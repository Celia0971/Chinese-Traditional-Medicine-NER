#/home/celia/Traditional_Chinese_Medicine_MyProject/Utils/model.py
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.layers.python.layers import initializers

from Utils.tf_utils.bert_modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint
from Utils.tf_utils import rnncell as rnn



class entity_model:
    def __init__(self, args):
        self.args = args

        # 喂入模型的数据占位符
        self.input_x_word = tf.placeholder(tf.int32, [None, None], name="input_x_word")
        self.input_x_len = tf.placeholder(tf.int32, name='input_x_len')
        self.input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')
        self.input_relation = tf.placeholder(tf.int32, [None, None], name='input_relation')  # 实体NER的真实标签
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        
        # BERT Embedding
        self.init_embedding(bert_init=True)
        output_layer = self.word_embedding

        # 超参数设置
        self.relation_num = self.args.relation_num
        self.initializer = initializers.xavier_initializer()
        self.lstm_dim = self.args.lstm_dim
        self.embed_dense_dim = self.args.embed_dense_dim
        self.dropout = self.args.dropout
        self.model_type = self.args.model_type
        
        # idcnn的超参数
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        self.filter_width = 3
        self.num_filter = self.lstm_dim
        self.embedding_dim = self.embed_dense_dim
        self.repeat_times = 4
        self.cnn_output_width = 0

        # CRF超参数
        used = tf.sign(tf.abs(self.input_x_word))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        print("lengths:" ,self.lengths ) 
        self.batch_size = tf.shape(self.input_x_word)[0]
        self.num_steps = tf.shape(self.input_x_word)[-1]
        if self.model_type == 'bilstm':#= ######1
            lstm_inputs = tf.nn.dropout(output_layer, self.dropout)
            lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)
            self.logits = self.project_layer(lstm_outputs)

        elif self.model_type == 'idcnn':#= ######2
            model_inputs = tf.nn.dropout(output_layer, self.dropout)
            model_outputs = self.IDCNN_layer(model_inputs)
            self.logits = self.project_layer_idcnn(model_outputs)
            
        else:
            raise KeyError

        # 计算损失
        self.loss = self.loss_layer(self.logits, self.lengths)
        

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.name_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.name_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.name_scope("project" if not name else name):
            with tf.name_scope("hidden"):
                outputdim = lstm_outputs.shape[2].value
                densedim = outputdim / 2
                W = tf.get_variable("HW", shape=[outputdim, densedim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("Hb", shape=[densedim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, outputdim])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.name_scope("logits"):
                W = tf.get_variable("LW", shape=[densedim, self.relation_num],
                                    dtype=tf.float32, initializer=self.initializer)
                
                b = tf.get_variable("Lb", shape=[self.relation_num], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                
                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.relation_num], name='pred_logits')

    def IDCNN_layer(self, model_inputs, name=None):
        """
        :param idcnn_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, cnn_output_width]
        """
        model_inputs = tf.expand_dims(model_inputs, 1)
        with tf.variable_scope("idcnn" if not name else name):
            shape = [1, self.filter_width, self.embedding_dim,
                     self.num_filter]
            print('shape==>',shape)
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim, self.num_filter],
                initializer=self.initializer
            )
            
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer")
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=tf.AUTO_REUSE):
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = tf.cond(self.is_training, lambda: 0.8, lambda: 1.0)
            # keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)

            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            return finalOut

    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.name_scope("project" if not name else name):
            # project to score of tags
            with tf.name_scope("logits"):
                W = tf.get_variable("PLW", shape=[self.cnn_output_width, self.relation_num],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("PLb", initializer=tf.constant(0.001, shape=[self.relation_num]))

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.relation_num], name='pred_logits')

    def adj_dice_loss(self, y_true, y_pred):
            probs = tf.nn.softmax(y_pred, axis=-1)
            probs = tf.gather(probs, indices=tf.expand_dims(y_true, axis=-1), axis=-1)
            probs_with_factor = ((1 - probs) ** 0.6) * probs
            loss = 1 - (2 * probs_with_factor + 0.4) / (probs_with_factor + 1 + 0.4)
            # loss = tf.reduce_mean(loss)
            return loss

    def loss_layer(self, project_logits, lengths, name=None):
        """
        计算CRF的loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.name_scope("crf_loss" if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.relation_num]), tf.zeros(shape=[self.batch_size, 1, 1])],
                axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.relation_num * tf.ones([self.batch_size, 1]), tf.int32), self.input_relation], axis=-1)
            
            """dice_loss"""
            dice_loss = self.adj_dice_loss(targets, logits)


            self.trans = tf.get_variable(
                name="transitions",
                shape=[self.relation_num + 1, self.relation_num + 1],  # 1 why +1? ########
                # shape=[self.relation_num, self.relation_num],  # 1
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                # tag_indices=self.input_relation,
                transition_params=self.trans,
                # sequence_lengths=lengths #15
                sequence_lengths=lengths + 1
            )  # + 1
            losses = -log_likelihood + self.args.diceloss_weight * dice_loss

            # print(f'log_likelihood loss: {-log_likelihood}') #Tensor("crf_loss/Neg_1:0", dtype=float32)
            # print(f'dice_loss: {dice_loss}') #Tensor("crf_loss/Mean:0", shape=(), dtype=float32)
            # Q: in dice_loss, is it necessary to use tf.reduce_mean()? 
            # Does it mean dice_loss through twice reduce_mean()? 

            # and if i annotate tf.reduce_mean() in dice_loss ,then print dice_loss as dice_loss: Tensor("crf_loss/sub_1:0", shape=(?, ?, ?, ?, 1), dtype=float32)

            return tf.reduce_mean(losses, name='loss')

    def init_embedding(self, bert_init=True):
        """
        对BERT的Embedding降维
        :param bert_init:
        :return:
        """
        with tf.name_scope('embedding'):
            word_embedding = self.bert_embed(bert_init)
            word_embedding = tf.layers.dense(word_embedding, self.args.embed_dense_dim, activation=tf.nn.relu)
            hidden_size = word_embedding.shape[-1].value
        self.word_embedding = word_embedding
        #print(word_embedding.shape)
        self.output_layer_hidden_size = hidden_size

    def bert_embed(self, bert_init=True):
        """ 读取BERT的TF模型
        :param bert_init:bool
        :return:
        """
        bert_config_file = self.args.bert_config_file
        bert_config = BertConfig.from_json_file(bert_config_file)
        model = BertModel(
            config=bert_config,
            is_training=self.is_training,  # 微调
            input_ids=self.input_x_word, #int32 Tensor of shape [batch_size, seq_length]
            input_mask=self.input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=False)
        
        print('model:',type(model))
        # """see see"""
        # print_txt = open(self.args.print_sth,"a+")
        # print('===model===\n', model, '\n', file=print_txt)
        # print_txt.close()
        
        ### 此处可修改 最后输出 在bert的源码里改？
        # 此处 加权融合？  什么意思？
        #means blend last numLayer=5 layers
        numLayer = self.args.num_layer
        for i in range(1,numLayer+1):
            if i==1:
                layer_logits = model.all_encoder_layers[-i]
            else:
                layer_logits += model.all_encoder_layers[-i]
        char_bert_outputs = layer_logits / numLayer
        
        if self.args.use_origin_bert:
            final_hidden_states = model.get_sequence_output()  # 原生bert
            self.args.embed_dense_dim = 768
        else:
            final_hidden_states = char_bert_outputs  # 多层融合bert
            self.args.embed_dense_dim = 512

        tvars = tf.trainable_variables()
        init_checkpoint = self.args.bert_file
        assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if bert_init:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        return final_hidden_states
    
    

    