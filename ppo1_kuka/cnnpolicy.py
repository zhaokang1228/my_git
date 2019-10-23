"""
CNN POLICY

图片输入大小　128*128

CNN 是一个两层的网络



"""
import gym
import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U
from baselines.common.distributions import make_pdtype

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_dims_p, hid_dims_v, train=True):
        assert isinstance(ob_space, gym.spaces.Box)
        self.pdtype = pdtype = make_pdtype(ac_space)
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.hid_dims_p = hid_dims_p
        self.hid_dims_v = hid_dims_v
        # # with tf.variable_scope('rms'):
        # #     self.ob_rms = RunningMeanStd(dtype=tf.float32, shape=ob_space.shape)
        # with tf.variable_scope('cnn'):
        #     self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))
        #     # self.obz = tf.clip_by_value((self.ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        #     self.x = self.ob/255.0
        #
        #     self.x = tf.nn.relu(U.conv2d(self.x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
        #     self.x = tf.nn.relu(U.conv2d(self.x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
        #     self.x = tf.nn.relu(U.conv2d(self.x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
        #     self.x = U.flattenallbut0(self.x)
        #     self.x = tf.nn.relu(tf.layers.dense(self.x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        #
        # odim = self.x.shape[-1]
        # odim = int(odim)
        # # print(self.ac_space.shape)
        # adim = pdtype.param_shape()[0]

        with tf.variable_scope("cnn"):
            self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))
            # self.obz = tf.clip_by_value((self.ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            self.x = self.ob / 255.0

            W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 4, 32], stddev=0.1))
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
            h_conv1 = tf.nn.relu(tf.nn.conv2d(self.x, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

            W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)
            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

            ##　全连接层，隐含层的节点个数为
            W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 1024], stddev=0.1))
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
            h_pool3_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
            # 将2D图像变成1D数据[n_samples,64,64,64]->>[n_samples,64*64]
            h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  # 非线性激活函数
            # h_fc1 = tf.matmul(h_pool3_flat, W_fc1) + b_fc1  # 非线性激活函数
            h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)  # 防止过拟合

            self.x = tf.nn.relu(tf.layers.dense(h_fc1_drop, 512, name="polfc",
                                                kernel_initializer=U.normc_initializer(1.0)))

        odim = self.x.shape[-1]
        odim = int(odim)
        # print(self.ac_space.shape)
        adim = pdtype.param_shape()[0]


        with tf.variable_scope('pol'):
            self._policy_nn(odim, adim, train)
        with tf.variable_scope('vf'):
            self._vf_nn(odim, adim, train)

    def _policy_nn(self, odim, adim, train):
        # activ = tf.nn.tanh
        # self.pdtype = make_pdtype(self.ac_space)
        # self.pdtype = DiagGaussianPdType(self.ac_space.shape[0])
        # hid1_size = 64
        # out = tf.layers.dense(self.x, adim, trainable=train,
        #                       kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1/adim)), name='out')
        #
        # self.pd = self.pdtype.pdfromflat(out)
        # self._act = U.function([self.ob], self.pd.sample()) # [self.pd.sample(), mean, logstd]
        # self .ac = self.pd.sample()
        # logits = tf.layers.dense(self.x, self.pdtype.param_shape()[0], name='logits',
        #                               kernel_initializer=U.normc_initializer(0.01))
        # self.pd = self.pdtype.pdfromflat(logits)
        mean = tf.layers.dense(self.x, self.pdtype.param_shape()[0] // 2, name="polfinal",
                               kernel_initializer=U.normc_initializer(0.01))
        logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[0] // 2],
                                 initializer=tf.zeros_initializer())
        # 链接
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        self.pd = self.pdtype.pdfromflat(pdparam)

        stochastic = U.get_placeholder(dtype=tf.bool, shape=(), name="stochastic")
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())

        self._act = U.function([stochastic, self.ob], ac)
        self.ac = ac

    def _vf_nn(self, odim, adim, train):
        activ = tf.nn.tanh
        # hid1_size = odim
        hid1_size = 128
        out = tf.layers.dense(self.x, 1, trainable=train,
              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / hid1_size)), name='output')
        # tf.squeeze()用于压缩张量中为1的轴
        self.vpred = tf.squeeze(out)
        self._val = U.function([self.ob], self.vpred)

    def act(self, ob,  stochastic=True):
        # ac1 = self._act(ob)
        if len(ob.shape) == 3: ob = ob[None]
        ac1 = self._act(stochastic, ob)
        # print("cnn_act",ac1[0])
        return ac1[0]

    def value(self, ob):
        if len(ob.shape) == 3: ob = ob[None]
        vpred1 = self._val(ob)
        return vpred1

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_policy_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope+'/pol')

    # cnn part is not trained in selection stage.
    def get_cnn_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope+'/cnn')

    def get_initial_state(self):
        return []


class CPolicy(object):
    def __init__(self, name, elitepi, *args, **kwargs):
        self.elitepi = elitepi
        self.adim = len(self.elitepi)
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, train=True):
        # with tf.variable_scope('rms'):
        #     self.ob_rms = RunningMeanStd(dtype=tf.float32, shape=ob_space.shape)
        # self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))
        # self.obz = tf.clip_by_value((self.ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        self.ob_space = ob_space
        with tf.variable_scope("cnn"):
            self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))
            # self.obz = tf.clip_by_value((self.ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            self.x = self.ob / 255.0

            W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 4, 32], stddev=0.1))
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
            h_conv1 = tf.nn.relu(tf.nn.conv2d(self.x, W_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

            W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)
            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

            ##　全连接层，隐含层的节点个数为
            W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 1024], stddev=0.1))
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
            h_pool3_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
            # 将2D图像变成1D数据[n_samples,64,64,64]->>[n_samples,64*64]
            h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  # 非线性激活函数
            # h_fc1 = tf.matmul(h_pool3_flat, W_fc1) + b_fc1  # 非线性激活函数
            h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)  # 防止过拟合

            self.x = tf.nn.relu(
                tf.layers.dense(h_fc1_drop, 512, name="polfc", kernel_initializer=U.normc_initializer(1.0)))

        with tf.variable_scope('selector'):
            self._policy_nn(int(self.x.shape[-1]), train)
        # with tf.variable_scope('vfun'):
        #     self._vf_nn(ob_space.shape[0], train)

    def _policy_nn(self, odim, train):
        activ = tf.nn.tanh
        # hid1_size = odim
        # hid3_size = self.adim * 5

        hid1_size = 16
        hid3_size = 16

        out = tf.layers.dense(self.x, hid1_size, trainable=train,
              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1/odim)), name="h1")
        # out = tf.layers.dense(out, hid3_size, activ, trainable=train,
        #       kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1/hid3_size)), name="h3")
        self.logits = tf.layers.dense(out, self.adim, activation=tf.nn.softmax, name='logits', trainable=train)

        self._act = U.function([self.ob], self.logits)

        # select_id = tf.argmax(self.logits,axis=1)
        # one_hot_actions = tf.one_hot(select_id, self.logits.get_shape().as_list()[-1])

        # for i in range(len(self.elitepi)):
        #     ac = tf.cond(select_id == i, lambda: self.elitepi[i].pd.sample(),lambda: ac)
            # self.elitepi[i].pd.sample() * tf.reshape(one_hot_actions[:, i], (-1, 1))

        # ac = tf.reduce_mean(acs, axis=0)
        # acs = [self.elitepi[i].pd.sample()*tf.reshape(self.logits[:,i],(-1,1))
        #                                             for i in range(len(self.elitepi))]
        # ac = tf.reduce_mean(acs,axis=0)
        # self._act = U.function([self.ob], [ac, self.logits])
        # u = tf.random_uniform(tf.shape(self.logits))
        # self.action = tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)
        # self._act = U.function([self.ob], self.action)

    def _vf_nn(self, odim, train):
        activ = tf.nn.tanh
        hid1_size = odim * 5
        hid3_size = 5
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        # out = tf.layers.dense(self.obz, hid1_size, activ,trainable=train,
        #       kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 /odim)), name="h1")
        out = tf.layers.dense(self.x, hid1_size, activ,trainable=train,
              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 /odim)), name="h1")
        out = tf.layers.dense(out, hid2_size, activ,trainable=train,
              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / hid1_size)), name="h2")
        out = tf.layers.dense(out, hid3_size, activ,trainable=train,
              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / hid2_size)), name="h3")
        out = tf.layers.dense(out, 1,trainable=train,
              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / hid3_size)), name='output')
        self.vpred = tf.squeeze(out)
        self._val = U.function([self.ob], self.vpred)

    def act(self, ob):
        if len(ob.shape) == 3: ob = ob[None]
        prob = self._act(ob)
        select_id = np.argmax(prob)
        ac1 = self.elitepi[select_id].act(ob)
        return ac1, prob

    def value(self, ob):
        vpred1 =  self._val(ob)
        return vpred1

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope+'/selector')

    def get_cnn_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope+'/cnn')


# class MLPDGPolicy(object):
#     """ NN-based policy approximation """
#     def __init__(self, obs_dim, act_dim,lr_, kl_targ,seed,noise_bias,elite_rate,popsize,num_worker,):
#         self.seed = seed
#         self.sigma_bias = noise_bias  # bias in stdev of output
#         self.num_worker = num_worker
#         self.popsize = popsize
#         self.elite_num = int(popsize*elite_rate)
#         self.kl_targ = kl_targ
#         self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
#         self.eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
#         self.epochs = 20
#         self.epochs1 = 20
#         self.lr_ = lr_ # dynamically adjust lr when D_KL out of control
#         self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#         self._build_graph()
#         self._init_session()
#
#     def _build_graph(self):
#         """ Build and initialize TensorFlow graph """
#         self.g = tf.Graph()
#         with self.g.as_default():
#             self._placeholders()
#             self._policy_nn()
#             self._logprob()
#             self._kl_entropy()
#             self._loss_train_op()
#             self.init = tf.global_variables_initializer()
#
#     def _init_session(self):
#         self.sess = tf.Session(graph=self.g, config=mp.tf_config)
#         self.sess.run(self.init)
#         self.variables.set_session(self.sess)
#
#     def _placeholders(self):
#         self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
#         self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
#         self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
#         # strength of D_KL loss terms:
#         self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
#         self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
#         # learning rate:
#         self.lr_ph = tf.placeholder(tf.float32, (), 'eta')
#         # log_vars and means with pi_old (previous step's policy parameters):
#         self.old_log_vars_ph = tf.placeholder(tf.float32, ( self.act_dim,), 'old_log_vars')
#         self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')
#
#         self.log_vars_sym = []
#         self.means_sym = []
#         for i in range(self.elite_num):
#             self.log_vars_sym.append(tf.placeholder(tf.float32, (self.act_dim,), 'log_vars%d'%i))
#             self.means_sym.append(tf.placeholder(tf.float32, (None, self.act_dim), 'means%d'%i))
#
#     def _policy_nn(self):
#         tf.set_random_seed(self.seed)
#         activ = tf.tanh
#         hid1_size = self.obs_dim * 10  # 10 empirically determined
#         hid3_size = self.act_dim * 10  # 10 empirically determined
#         hid2_size = int(np.sqrt(hid1_size * hid3_size))
#
#         self.lr = self.lr_ / np.sqrt(hid1_size)  # 9e-4 empirically determined
#         self.lr_m = 1e-4 / np.sqrt(hid1_size)
#
#         if mp.rank == 0:
#             print('Policy Fun -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
#                   .format(hid1_size, hid2_size, hid3_size, self.lr))
#
#         h1 = activ(fc(self.obs_ph, 'h1', nh=hid1_size, init_scale=np.sqrt(2)))
#         h2 = activ(fc(h1, 'h2', nh=hid2_size, init_scale=np.sqrt(2)))
#         h3 = activ(fc(h2, 'h3', nh=hid3_size, init_scale=np.sqrt(2)))
#         self.means = fc(h3, 'means', nh=self.act_dim, init_scale=np.sqrt(2))
#
#         logvar_speed = (10 * hid3_size) // 48
#         log_vars=tf.get_variable('logvars',(logvar_speed,self.act_dim),
#                                  tf.float32,tf.constant_initializer(0.0))
#         self.log_vars = tf.reduce_sum(log_vars, axis=0)-1
#         out_std = tf.exp(0.5 * self.log_vars + self.sigma_bias)
#         output_noise = tf.random_normal(shape=(self.act_dim,), dtype=tf.float32) * out_std
#         self.action = output_noise + self.means
#
#         self.variables = TFVariables(self.action)
#         self.num_params = sum([np.prod(variable.shape.as_list())
#                                 for _, variable in self.variables.variables.items()])
#
#     def _logprob(self):
#         logp = -0.5 * tf.reduce_sum(self.log_vars)
#         logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
#                                      tf.exp(self.log_vars), axis=1)
#         self.logp = logp
#
#         logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
#         logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
#                                          tf.exp(self.old_log_vars_ph), axis=1)
#         self.logp_old = logp_old
#
#     def _kl_entropy(self):
#         log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
#         log_det_cov_new = tf.reduce_sum(self.log_vars)
#         tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))
#
#         self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
#                                        tf.reduce_sum(tf.square(self.means - self.old_means_ph) /
#                                                      tf.exp(self.log_vars), axis=1) -
#                                        self.act_dim)
#         self.entropy = 0.5*(self.act_dim*(np.log(2*np.pi)+1)+tf.reduce_sum(self.log_vars))
#
#         self.kl_sym = 0
#         for i in range(self.elite_num):
#             log_det_cov_old = tf.reduce_sum(self.log_vars_sym[i])
#             tr_old_new = tf.reduce_sum(tf.exp(self.log_vars_sym[i] - self.log_vars))
#             self.kl_sym+=0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
#                                        tf.reduce_sum(tf.square(self.means - self.means_sym[i]) /
#                                                      tf.exp(self.log_vars), axis=1) -
#                                        self.act_dim)
#         self.kl_sym = self.kl_sym / self.elite_num
#
#     def _loss_train_op(self):
#         loss1 = -tf.reduce_mean(self.advantages_ph * tf.exp(self.logp - self.logp_old))
#         loss2 = tf.reduce_mean(self.beta_ph * self.kl)
#         loss3 = self.eta_ph*tf.square(tf.maximum(np.array(0).astype(np.float32),
#                                                    self.kl-2.0*self.kl_targ))
#         self.loss_s = loss1  + loss2 + loss3
#         self.optimizer = tf.train.AdamOptimizer(self.lr_ph)
#         self.train_op_s = self.optimizer.minimize(self.loss_s)
#
#         self.loss_m = tf.reduce_mean(self.kl_sym)
#         self.optimizer_m = tf.train.AdamOptimizer(self.lr_m)
#         self.train_op_m = self.optimizer_m.minimize(self.loss_m)
#
#
#     def set_policy_params(self, policy_params):
#         self.variables.set_flat(policy_params)
#     def get_policy_params(self):
#         policy_params = self.variables.get_flat()
#         return policy_params
#
#     def get_action(self, obs):
#         if np.ndim(obs) == 1:
#             obs=obs[np.newaxis,:]
#         feed_dict = {self.obs_ph: obs}
#         return self.sess.run(self.action, feed_dict=feed_dict)
#
#     def pg_update(self, observes, actions, advantages):
#         feed_dict = {self.obs_ph: observes,
#                      self.act_ph: actions,
#                      self.advantages_ph: advantages,
#                      self.beta_ph: self.beta,
#                      self.eta_ph: self.eta,
#                      self.lr_ph: self.lr * self.lr_multiplier}
#         old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars], feed_dict)
#         feed_dict[self.old_means_ph] = old_means_np
#         feed_dict[self.old_log_vars_ph] = old_log_vars_np
#         kl = 0
#         for e in range(self.epochs):
#             self.sess.run(self.train_op_s, feed_dict)
#             kl = self.sess.run(self.kl, feed_dict)
#             if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
#                 break
#
#         if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
#             self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
#             if self.beta > 30 and self.lr_multiplier > 0.1:
#                 self.lr_multiplier /= 1.5
#         elif kl < self.kl_targ / 2:
#             self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
#             if self.beta < (1 / 30) and self.lr_multiplier < 10:
#                 self.lr_multiplier *= 1.5
#
#         return self.get_policy_params()
#
#     def combine(self, observes, actions, elite_p):
#         feed_dict = {self.obs_ph: observes, self.act_ph: actions,}
#         for i in range(self.elite_num):
#             self.set_policy_params(np.array(elite_p[i]))
#             means, log_vars = self.sess.run([self.means, self.log_vars], feed_dict)
#             feed_dict[self.means_sym[i]] = means
#             feed_dict[self.log_vars_sym[i]] = log_vars
#
#         mean_policy = np.mean(elite_p, axis=0)
#         self.set_policy_params(mean_policy)
#         for e in range(self.epochs1):
#             self.sess.run(self.train_op_m, feed_dict)
#         # kl = self.sess.run(self.kl_sym, feed_dict)
#
#     def close_sess(self):
#         self.sess.close()
#
#
# class MLPCPolicy(object):
#     """ NN-based policy approximation """
#     def __init__(self, obs_dim, act_dim,lr_, kl_targ,seed,noise_bias,elite_rate,popsize,num_worker,):
#         """
#             obs_dim: num observation dimensions (int)
#             act_dim: num action dimensions (int)
#             kl_targ: target KL divergence between pi_old and pi_new
#         """
#         self.seed = seed
#         self.sigma_bias = noise_bias  # bias in stdev of output
#         self.num_worker = num_worker
#         self.popsize = popsize
#         self.elite_num = int(popsize*elite_rate)
#         self.kl_targ = kl_targ
#         self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
#         self.eta = 50  # multiplier for D_KL-kl_targ hinge-squared loss
#         self.epochs = 40
#         self.epochs1 = 10
#         self.lr_ = lr_ # dynamically adjust lr when D_KL out of control
#         self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#         self._build_graph()
#         self._init_session()
#
#     def _build_graph(self):
#         """ Build and initialize TensorFlow graph """
#         self.g = tf.Graph()
#         with self.g.as_default():
#             self._placeholders()
#             self._policy_nn()
#             self._logprob()
#             self._kl_entropy()
#             self._loss_train_op()
#             self.init = tf.global_variables_initializer()
#
#     def _init_session(self):
#         """Launch TensorFlow session and initialize variables"""
#         self.sess = tf.Session(graph=self.g, config=mp.tf_config)
#         self.sess.run(self.init)
#         self.variables.set_session(self.sess)
#
#     def _placeholders(self):
#         """ Input placeholders"""
#         self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
#         self.act_ph = tf.placeholder(tf.int32, (None), 'act')
#         self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
#         # strength of D_KL loss terms:
#         self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
#         self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
#         # learning rate:
#         self.lr_ph = tf.placeholder(tf.float32, (), 'eta')
#         self.logits_old = tf.placeholder(tf.float32, (None, self.act_dim), 'old_logits')
#
#         self.logits_sym = []
#         for i in range(self.elite_num):
#             self.logits_sym.append(tf.placeholder(tf.float32, (None, self.act_dim), 'logits%d'%i))
#
#     def _policy_nn(self):
#         """ Neural net for policy approximation function
#         Policy parameterized by Gaussian means and variances. NN outputs mean
#          action based on observation. Trainable variables hold log-variances
#          for each action dimension (i.e. variances not determined by NN).
#         """
#         tf.set_random_seed(self.seed)
#         activ = tf.tanh
#         # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
#         hid1_size = self.obs_dim * 10  # 10 empirically determined
#         hid3_size = self.act_dim * 10  # 10 empirically determined
#         hid2_size = int(np.sqrt(hid1_size * hid3_size))
#         self.lr = self.lr_ / np.sqrt(hid1_size)
#         self.lr_m = 1e-4 / np.sqrt(hid1_size)
#
#         if mp.rank == 0:
#             print('Policy Fun -- h1: {}, h2: {}, h3: {}, lr: {:.3g}'
#                   .format(hid1_size, hid2_size, hid3_size, self.lr))
#
#         # 3 hidden layers with tanh activations
#         h1 = activ(fc(self.obs_ph, 'h1', nh = hid1_size, init_scale=np.sqrt(2)))
#         h2 = activ(fc(h1, 'h2', nh = hid2_size, init_scale=np.sqrt(2)))
#         h3 = activ(fc(h2, 'h3', nh = hid3_size, init_scale=np.sqrt(2)))
#         self.logits = fc(h3, 'logits', nh = self.act_dim)
#
#         self.variables = TFVariables(self.logits)
#         self.num_params = sum([np.prod(variable.shape.as_list())
#                                 for _, variable in self.variables.variables.items()])
#
#         u = tf.random_uniform(tf.shape(self.logits))
#         self.action = tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)
#
#     def _logprob(self):
#         """ Calculate log probabilities of a batch of observations & actions
#         Calculates log probabilities using previous step's model parameters and
#         new parameters being trained.
#         """
#         self.neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
#                                                                       labels=self.act_ph)
#
#         self.neglogp_old = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_old,
#                                                                       labels=self.act_ph)
#
#     def _kl_entropy(self):
#         """
#             1. KL divergence between old and new distributions
#             2. Entropy of present policy given states and actions
#         """
#         a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
#         a1 = self.logits_old - tf.reduce_max(self.logits_old, axis=-1, keep_dims=True)
#         ea0 = tf.exp(a0)
#         ea1 = tf.exp(a1)
#         z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
#         z1 = tf.reduce_sum(ea1, axis=-1, keep_dims=True)
#         p0 = ea0 / z0
#
#         self.kl = .5 * tf.reduce_mean(tf.square(self.neglogp - self.neglogp_old))
#         # self.kl = tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)
#         self.entropy = tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
#
#         self.kl_sym = 0
#         for i in range(self.elite_num):
#             neglogp_old = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_sym[i],
#                                                                          labels=self.act_ph)
#             self.kl_sym+=0.5 * tf.reduce_mean(tf.square(self.neglogp - neglogp_old))
#         self.kl_sym = self.kl_sym/(self.elite_num)
#
#     def _loss_train_op(self):
#         """
#         Three loss terms:
#             1) standard policy gradient
#             2) D_KL(pi_old || pi_new)
#             3) Hinge loss on [D_KL - kl_targ]^2
#         See: https://arxiv.org/pdf/1707.02286.pdf
#         """
#         loss1 = -tf.reduce_mean(self.advantages_ph*tf.exp(self.neglogp_old-self.neglogp))
#         loss2 = tf.reduce_mean(self.beta_ph * self.kl)
#         loss3 = self.eta_ph * tf.square(tf.maximum(np.array(0).astype(np.float32),
#                                                    self.kl-2.0*self.kl_targ))
#         self.loss_s = loss1  + loss2 + loss3
#         self.optimizer = tf.train.AdamOptimizer(self.lr_ph)
#         self.train_op_s = self.optimizer.minimize(self.loss_s)
#
#         self.loss_m = tf.reduce_mean(self.kl_sym)
#         self.optimizer_m = tf.train.AdamOptimizer(self.lr_m)
#         self.train_op_m = self.optimizer_m.minimize(self.loss_m)
#
#
#     def get_action(self, obs):
#         if np.ndim(obs) == 1:
#             obs=obs[np.newaxis,:]
#         """Draw sample from policy distribution"""
#         feed_dict = {self.obs_ph: obs}
#         a = self.sess.run(self.action, feed_dict=feed_dict)[0]
#         return a
#
#     def set_policy_params(self, policy_params):
#         self.variables.set_flat(policy_params)
#     def get_policy_params(self):
#         policy_params = self.variables.get_flat()
#         return policy_params
#
#     def pg_update(self, observes, actions, advantages):
#         feed_dict = {self.obs_ph: observes,
#                      self.act_ph: actions,
#                      self.advantages_ph: advantages,
#                      self.beta_ph: self.beta,
#                      self.eta_ph: self.eta,
#                      self.lr_ph: self.lr * self.lr_multiplier}
#         logits_old = self.sess.run(self.logits, feed_dict)
#         feed_dict[self.logits_old] = logits_old
#         loss, kl, entropy = 0, 0, 0
#         for e in range(self.epochs):
#             self.sess.run(self.train_op_s, feed_dict)
#             loss, kl, entropy = self.sess.run([self.loss_s, self.kl, self.entropy], feed_dict)
#             if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
#                 break
#         if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
#             self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
#             if self.beta > 30 and self.lr_multiplier > 0.1:
#                 self.lr_multiplier /= 1.5
#         elif kl < self.kl_targ / 2:
#             self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
#             if self.beta < (1 / 30) and self.lr_multiplier < 10:
#                 self.lr_multiplier *= 1.5
#         new_policy_params = self.get_policy_params()
#         return new_policy_params
#
#     def combine(self, observes, actions, elite_p):
#         feed_dict = {self.obs_ph: observes,
#                      self.act_ph: actions,}
#         for i in range(self.elite_num):
#             self.set_policy_params(np.array(elite_p[i]))
#             logits = self.sess.run(self.logits, feed_dict)
#             feed_dict[self.logits_sym[i]] = logits
#
#         mean_policy = np.mean(elite_p, axis=0)
#         self.set_policy_params(mean_policy)
#         for e in range(self.epochs1):
#             self.sess.run(self.train_op_m, feed_dict)
#         # kl = self.sess.run(self.kl_sym, feed_dict)
#
#     def close_sess(self):
#         """ Close TensorFlow session """
#         self.sess.close()