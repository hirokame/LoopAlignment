import numpy as np
class Whittington_Model:
  def __init__(self, x_dim, num_batch, lr_x, lr_weight,f):
      self.lr_x   = lr_x
      self.lr_weight   = lr_weight
      self.x_dim = x_dim
      self.num_module  = len(x_dim)
      self.num_batch   = num_batch
      self.f = f

      self.initialize_states() # 活動の初期化
      self.initialize_weight() # 重みの初期化

  def initialize_states(self):
    self.x   = []
    self.e   = []
    for i in range(self.num_module):
        self.x.append(np.zeros((self.num_batch,self.x_dim[i])))
    for i in range(self.num_module-1):
        self.e.append(np.zeros((self.num_batch,self.x_dim[i+1])))

  def initialize_weight(self):
    self.U      = []
    self.b      = []
    for i in range(self.num_module-1):
      limit,bias = self.f.init_weight(self.x_dim[i],self.x_dim[i+1])
      self.b.append(bias)
      self.U.append(np.random.uniform(low=-limit, high=limit, size=(self.x_dim[i],self.x_dim[i+1])))

  def calculate_energy(self):
      energy = sum(np.sum(ei**2) for ei in self.e) / self.num_batch
      return energy

  def save_cossim(self):
    cossim = 1
    return cossim

  def update_x(self, inputs, target, num_loop):
      lr_x=self.lr_x
      self.x[0] = inputs
      self.x[-1] = target

      # errorの更新
      for i in range(self.num_module - 1):
          self.e[i] = self.x[i+1] - (self.f(self.x[i]) @ self.U[i]) - self.b[i]

      # F_tの更新
      F_t = -np.sum([np.sum(np.square(ei)) for ei in self.e])

      for _ in range(num_loop):
        F_prev = F_t.copy()
        #xの更新
        for i in range(1,self.num_module - 1):
          dx = self.e[i-1] - (self.e[i] @ self.U[i].T) * self.f.deriv(self.x[i])
          self.x[i] -= lr_x * dx

        # errorの更新
        for i in range(self.num_module - 1):
          self.e[i] = self.x[i+1] - (self.f(self.x[i]) @ self.U[i]) - self.b[i]

        F_t = -np.sum([np.sum(np.square(ei)) for ei in self.e]) # F_tの更新
        diff = F_t - F_prev #errorの減少量を計算

        # errorの減少量が十分に小さい場合はスキップ
        if diff < 0:
            lr_x /= 2
        elif abs(diff) < 1e-6:
            break

  def update_weight(self):# 重みの更新
    db=[]
    dU=[]
    # U,bの更新
    for i in range(self.num_module - 1):
        db      .append(np.sum(self.e[i], axis=0) / self.num_batch)
        dU      .append(self.f(self.x[i]).T @ self.e[i] / self.num_batch)
        self.U[i] += self.lr_weight * dU[i]
        self.b[i] += self.lr_weight * db[i]

  def predict(self, input):
    # 各レイヤーの更新
    self.x[0]=input
    for i in range(self.num_module-1):
      self.x[i+1]=np.dot(self.f(self.x[i]),self.U[i])+self.b[i]                      # (batch_size,細胞数)
    return self.x[-1]

class LA_model(Whittington_Model):

  def initialize_weight(self):
    self.U = []
    self.U_FB = []
    self.b = []
    for i in range(self.num_module-1):
      limit,bias = self.f.init_weight(self.x_dim[i],self.x_dim[i+1])
      self.b.append(bias)
      self.U.append(np.random.uniform(low=-limit, high=limit, size=(self.x_dim[i],self.x_dim[i+1])))
      self.U_FB.append(np.random.uniform(low=-limit, high=limit, size=(self.x_dim[i],self.x_dim[i+1])))

  def initialize_weight_uni(self, mean, std):
    self.U = []
    self.U_FB = []
    self.b = []
    for i in range(self.num_module-1):
      limit,bias = self.f.init_weight(self.x_dim[i],self.x_dim[i+1])
      high = mean + np.sqrt(3) * std
      low  = mean - np.sqrt(3) * std
      self.b.append(bias)
      self.U.append(np.random.uniform(low=-low, high=high, size=(self.x_dim[i],self.x_dim[i+1])))
      self.U_FB.append(np.random.uniform(low=-low, high=high, size=(self.x_dim[i],self.x_dim[i+1])))

  def initialize_weight_nor(self, mean, std):
    self.U = []
    self.U_FB = []
    self.b = []
    for i in range(self.num_module-1):
      limit,bias = self.f.init_weight(self.x_dim[i],self.x_dim[i+1])
      self.b.append(bias)
      self.U.append(np.random.normal(loc=mean, scale=std, size=(self.x_dim[i],self.x_dim[i+1])))
      self.U_FB.append(np.random.normal(loc=mean, scale=std, size=(self.x_dim[i],self.x_dim[i+1])))

  def save_cossim(self):
    U      = np.array(self.U[1])     .reshape(-1)
    U_FB = np.array(self.U_FB[1]).reshape(-1)
    U_norm      = U      / (np.linalg.norm(U     , axis=-1, keepdims=True) + 1e-10)
    U_FB_norm = U_FB / (np.linalg.norm(U_FB, axis=-1, keepdims=True) + 1e-10)
    cossim = np.sum(U_norm * U_FB_norm, axis=-1)
    cossim = np.clip(cossim, -1.0, 1.0)
    return cossim

  def update_x(self, inputs, target, num_loop):
    lr_x=self.lr_x
    self.x[0] = inputs
    self.x[-1] = target

    # errorの更新
    for i in range(self.num_module - 1):
      self.e[i] = self.x[i+1] - (self.f(self.x[i]) @ self.U[i]) - self.b[i]

    # F_tの更新
    F_t = -np.sum([np.sum(np.square(ei)) for ei in self.e])

    for _ in range(num_loop):
      F_prev = F_t.copy()
      
      # xの更新
      for i in range(1,self.num_module - 1):
        dx = self.e[i-1] - (self.e[i] @ self.U_FB[i].T) * self.f.deriv(self.x[i])
        self.x[i] -= lr_x * dx

      # errorの更新
      for i in range(self.num_module - 1):
        self.e[i] = self.x[i+1] - (self.f(self.x[i]) @ self.U[i]) - self.b[i]

      F_t = -np.sum([np.sum(np.square(ei)) for ei in self.e]) # F_tの更新
      diff = F_t - F_prev #errorの減少量を計算

      # errorの減少量が十分に小さい場合はスキップ
      if diff < 0:
          lr_x /= 2
      elif abs(diff) < 1e-6:
          break
      
  def update_weight_fixW0(self): # 重みの更新(W0は固定)
    #U,bの更新 
    for i in range(1,self.num_module - 1):
        db = np.sum(self.e[i], axis=0) / self.num_batch
        dU = self.f(self.x[i]).T @ self.e[i] / self.num_batch
        self.U[i] += self.lr_weight * dU
        self.b[i] += self.lr_weight * db

  # noiseを与える処理
  def add_noise_FF(self, n_rate, ratio=1):
    for i in range(len(self.U)):
        matrix = self.U[i]
        num_cols = matrix.shape[1]

        # ノイズを加える列のインデックスをランダムに選ぶ（全体の10%など）
        num_to_modify = max(1, int(1 * num_cols))
        cols_to_modify = np.random.choice(num_cols, size=num_to_modify, replace=False)

        for col_idx in cols_to_modify:
            # noise_width=np.sqrt(1 / (self.cortex_cell[i] + self.num_cell[i]))*0.2*0.998**(self.step_count)
            noise_width=np.sqrt(1 / (self.x_dim[i] + self.x_dim[i+1]))*n_rate*ratio
            noise = np.random.uniform(low=-noise_width, high=noise_width , size=matrix[:, col_idx].shape)
            matrix[:, col_idx] += noise
            matrix[:, col_idx] /= np.linalg.norm(matrix[:, col_idx]) + 1e-8
        self.U[i] = matrix

  def add_noise_FB(self, n_rate):
    for i in range(len(self.U_FB)):
        matrix = self.U_FB[i]
        num_cols = matrix.shape[1]

        # ノイズを加える列のインデックスをランダムに選ぶ（全体の10%など）
        num_to_modify = max(1, int(1 * num_cols))
        cols_to_modify = np.random.choice(num_cols, size=num_to_modify, replace=False)

        for col_idx in cols_to_modify:
            # noise_width=np.sqrt(1 / (self.cortex_cell[i] + self.num_cell[i]))*0.2*0.998**(self.step_count)
            noise_width=np.sqrt(1 / (self.x_dim[i] + self.x_dim[i+1]))*n_rate
            noise = np.random.uniform(low=-noise_width, high=noise_width , size=matrix[:, col_idx].shape)
            matrix[:, col_idx] += noise
            matrix[:, col_idx] /= np.linalg.norm(matrix[:, col_idx]) + 1e-8
        self.U_FB[i] = matrix

class PCRHL(Whittington_Model):
  def initialize_weight(self):
    self.U = []
    self.U_FB = []
    self.b = []
    for i in range(self.num_module-1):
      limit,bias = self.f.init_weight(self.x_dim[i],self.x_dim[i+1])
      self.b.append(bias)
      self.U.append(np.random.uniform(low=-limit, high=limit, size=(self.x_dim[i],self.x_dim[i+1])))
      self.U_FB.append(np.random.uniform(low=-limit, high=limit, size=(self.x_dim[i],self.x_dim[i+1])))

  def save_cossim(self):
    U      = np.array(self.U[1])     .reshape(-1)
    U_FB = np.array(self.U_FB[1]).reshape(-1)
    U_norm      = U      / (np.linalg.norm(U     , axis=-1, keepdims=True) + 1e-10)
    U_FB_norm = U_FB / (np.linalg.norm(U_FB, axis=-1, keepdims=True) + 1e-10)
    cossim = np.sum(U_norm * U_FB_norm, axis=-1)
    cossim = np.clip(cossim, -1.0, 1.0)
    return cossim
  
  def update_x(self, inputs, target, num_loop):
    lr_x=self.lr_x
    self.x[0] = inputs
    self.x[-1] = target

    #errorの更新
    for i in range(self.num_module - 1):
      self.e[i] = self.x[i+1] - (self.f(self.x[i]) @ self.U[i]) - self.b[i]

    #F_tの更新
    F_t = -np.sum([np.sum(np.square(ei)) for ei in self.e])

    for _ in range(num_loop):
      F_prev = F_t.copy()
      #xの更新
      for i in range(1,self.num_module - 1):
        dx = self.e[i-1] - (self.e[i] @ self.U_FB[i].T) * self.f.deriv(self.x[i])
        self.x[i] -= lr_x * dx

      #errorの更新
      for i in range(self.num_module - 1):
        self.e[i] = self.x[i+1] - (self.f(self.x[i]) @ self.U[i]) - self.b[i]

      F_t = -np.sum([np.sum(np.square(ei)) for ei in self.e]) #F_tの更新
      diff = F_t - F_prev #errorの減少量を計算

      #errorの減少量が十分に小さい場合はスキップ
      if diff < 0:
          lr_x /= 2
      elif abs(diff) < 1e-6:
          break
  def update_weight(self):#重みの更新
    db=[]
    dU=[]
    dU_rand = []
    #U,bの更新
    for i in range(self.num_module - 1):
        db      .append(np.sum(self.e[i], axis=0) / self.num_batch)
        dU      .append(self.f(self.x[i]).T @ self.e[i] / self.num_batch)
        dU_rand.append((self.f(self.e[i]).T @ self.x[i]).T / self.num_batch)
        self.U[i] += self.lr_weight * dU[i]
        self.U_FB[i] += self.lr_weight * dU_rand[i]
        self.b[i] += self.lr_weight * db[i]

  # noiseを与える処理
  def add_noise_FF(self, n_rate, ratio=1):
    for i in range(len(self.U)):
        matrix = self.U[i]
        num_cols = matrix.shape[1]

        # ノイズを加える列のインデックスをランダムに選ぶ（全体の10%など）
        num_to_modify = max(1, int(1 * num_cols))
        cols_to_modify = np.random.choice(num_cols, size=num_to_modify, replace=False)

        for col_idx in cols_to_modify:
            # noise_width=np.sqrt(1 / (self.cortex_cell[i] + self.num_cell[i]))*0.2*0.998**(self.step_count)
            noise_width=np.sqrt(1 / (self.x_dim[i] + self.x_dim[i+1]))*n_rate*ratio
            noise = np.random.uniform(low=-noise_width, high=noise_width , size=matrix[:, col_idx].shape)
            matrix[:, col_idx] += noise
            matrix[:, col_idx] /= np.linalg.norm(matrix[:, col_idx]) + 1e-8
        self.U[i] = matrix

  def add_noise_FB(self, n_rate):
    for i in range(len(self.U_FB)):
        matrix = self.U_FB[i]
        num_cols = matrix.shape[1]

        # ノイズを加える列のインデックスをランダムに選ぶ（全体の10%など）
        num_to_modify = max(1, int(1 * num_cols))
        cols_to_modify = np.random.choice(num_cols, size=num_to_modify, replace=False)

        for col_idx in cols_to_modify:
            # noise_width=np.sqrt(1 / (self.cortex_cell[i] + self.num_cell[i]))*0.2*0.998**(self.step_count)
            noise_width=np.sqrt(1 / (self.x_dim[i] + self.x_dim[i+1]))*n_rate
            noise = np.random.uniform(low=-noise_width, high=noise_width , size=matrix[:, col_idx].shape)
            matrix[:, col_idx] += noise
            matrix[:, col_idx] /= np.linalg.norm(matrix[:, col_idx]) + 1e-8
        self.U_FB[i] = matrix

class dual_LA_model(Whittington_Model):
  def __init__(self, x1_dim, x2_dim, num_batch, lr_x, lr_weight, f):
      self.lr_x   = lr_x
      self.lr_weight   = lr_weight
      self.x1_dim = x1_dim
      self.x2_dim = x2_dim
      self.num_module  = len(x1_dim)
      self.num_batch   = num_batch
      self.f = f

      self.initialize_states() # 活動の初期化
      self.initialize_weight() # 重みの初期化
  
  def initialize_states(self):
    self.x_PC1   = []
    self.x_PC2   = []
    self.e_PC1   = []
    self.e_PC2   = []
    for i in range(self.num_module):
        self.x_PC1.append(np.zeros((self.num_batch,self.x1_dim[i])))
        self.x_PC2.append(np.zeros((self.num_batch,self.x2_dim[i])))
    for i in range(self.num_module-1):
        self.e_PC1.append(np.zeros((self.num_batch,self.x1_dim[i+1])))
        self.e_PC2.append(np.zeros((self.num_batch,self.x2_dim[i+1])))  

  def initialize_weight(self):
    self.U = []

    limit,bias = self.f.init_weight(self.x1_dim[0],self.x1_dim[1])
    self.U.append(np.random.uniform(low=-limit, high=limit, size=(self.x1_dim[0],self.x1_dim[1])))
    self.U.append(np.random.uniform(low=-limit, high=limit, size=(self.x2_dim[0],self.x2_dim[1])))

    limit,bias = self.f.init_weight(self.x1_dim[1],self.x1_dim[2])
    self.U.append(np.random.uniform(low=-limit, high=limit, size=(self.x1_dim[1],self.x1_dim[2])))
    self.U.append(np.random.uniform(low=-limit, high=limit, size=(self.x1_dim[2],self.x2_dim[1])))
    self.U.append(np.random.uniform(low=-limit, high=limit, size=(self.x2_dim[1],self.x2_dim[2])))
    self.U.append(np.random.uniform(low=-limit, high=limit, size=(self.x2_dim[2],self.x1_dim[1])))

  def calculate_energy1(self):
    energy = sum(np.sum(ei**2) for ei in self.e_PC1) / self.num_batch
    return energy

  def calculate_energy2(self):
    energy = sum(np.sum(ei**2) for ei in self.e_PC2) / self.num_batch
    return energy

  def save_cossim1(self):
      # U_PC1[1] と PC2の行列積を取得
      U   = np.array(self.U[2].T).reshape(-1)
      U_FB   = np.array(self.U[3] @ self.U[4] @ self.U[5]).reshape(-1)

      # 正規化
      U_norm = U / (np.linalg.norm(U) + 1e-10)
      U_FB_norm = U_FB / (np.linalg.norm(U_FB) + 1e-10)

      # コサイン類似度
      cos_sim = np.sum(U_norm * U_FB_norm)
      cos_sim = np.clip(cos_sim, -1.0, 1.0)

      return cos_sim

  def save_cossim2(self):
      U   = np.array(self.U[4].T).reshape(-1)
      U_FB   = np.array(self.U[5] @ self.U[2] @ self.U[3]).reshape(-1)

      U_norm = U / (np.linalg.norm(U) + 1e-10)
      U_FB_norm = U_FB / (np.linalg.norm(U_FB) + 1e-10)

      cos_sim = np.sum(U_norm * U_FB_norm)
      cos_sim = np.clip(cos_sim, -1.0, 1.0)

      return cos_sim
  
  # PC1 Learn
  def update_x1(self, inputs, target, num_loop):
    self.x_PC1[0 ] = inputs
    self.x_PC1[-1] = target

    # PC1 errorの更新
    self.e_PC1[0] = self.x_PC1[1] - (self.x_PC1[0] @ self.U[0])
    self.e_PC1[1] = self.x_PC1[2] - (self.x_PC1[1] @ self.U[2])

    for _ite in range(num_loop):
      self.x_PC2[1] = self.e_PC1[1] @ self.U[3]
      self.e_PC2[1] = self.x_PC2[1] @ self.U[4]
      dx = self.e_PC1[0] - (self.e_PC2[1] @ self.U[5])
      self.x_PC1[1] -= self.lr_x * dx

      self.e_PC1[0] = self.x_PC1[1] - (self.x_PC1[0] @ self.U[0])
      self.e_PC1[1] = self.x_PC1[2] - (self.x_PC1[1] @ self.U[2])

  def update_weight1(self) :
    dU=[]
    for i in range(self.num_module - 1):
      dU.append(self.x_PC1[i].T @ self.e_PC1[i] / self.num_batch)
    self.U[0] += self.lr_weight * dU[0]
    self.U[2] += self.lr_weight * dU[1]

  # PC2 Learn
  def update_x2(self, inputs, target, num_loop):
    self.x_PC2[0 ] = inputs
    self.x_PC2[-1] = target

    # PC2 errorの更新
    self.e_PC2[0] = self.x_PC2[1] - (self.x_PC2[0] @ self.U[1])
    self.e_PC2[1] = self.x_PC2[2] - (self.x_PC2[1] @ self.U[4])

    for _ite in range(num_loop):
      self.x_PC1[1] = self.e_PC2[1] @ self.U[5]
      self.e_PC1[1] = self.x_PC1[1] @ self.U[2]
      dx = self.e_PC2[0] - (self.e_PC1[1] @ self.U[3])
      self.x_PC2[1] -= self.lr_x * dx

      self.e_PC2[0] = self.x_PC2[1] - (self.x_PC2[0] @ self.U[1])
      self.e_PC2[1] = self.x_PC2[2] - (self.x_PC2[1] @ self.U[4])

  def update_weight2(self) :
    dU=[]
    for i in range(self.num_module - 1):
      dU.append(self.x_PC2[i].T @ self.e_PC2[i] / self.num_batch)
    self.U[1] += self.lr_weight * dU[0]
    self.U[4] += self.lr_weight * dU[1]

  def predict1(self, input):
    self.x_PC1[0] = input
    self.x_PC1[1] = self.x_PC1[0] @ self.U[0]
    self.x_PC1[2] = self.x_PC1[1] @ self.U[2]
    return self.x_PC1[-1]

  def predict2(self, input):
    self.x_PC2[0] = input
    self.x_PC2[1] = self.x_PC2[0] @ self.U[1]
    self.x_PC2[2] = self.x_PC2[1] @ self.U[4]
    return self.x_PC2[-1]

  # noiseを与える処理
  def add_noise_FF(self, n_rate):
    matrix = self.U[2]
    num_cols = matrix.shape[1]

    # ノイズを加える列のインデックスをランダムに選ぶ（全体の10%など）
    num_to_modify = max(1, int(1 * num_cols))
    cols_to_modify = np.random.choice(num_cols, size=num_to_modify, replace=False)

    for col_idx in cols_to_modify:
        # noise_width=np.sqrt(1 / (self.cortex_cell[i] + self.num_cell[i]))*0.2*0.998**(self.step_count)
        noise_width=np.sqrt(1 / (self.x1_dim[2] + self.x2_dim[1]))*n_rate
        noise = np.random.uniform(low=-noise_width, high=noise_width , size=matrix[:, col_idx].shape)
        matrix[:, col_idx] += noise
        matrix[:, col_idx] /= np.linalg.norm(matrix[:, col_idx]) + 1e-8
    self.U[2] = matrix

  def add_noise_FB(self, n_rate):
    matrix = self.U[4]
    num_cols = matrix.shape[1]

    # ノイズを加える列のインデックスをランダムに選ぶ（全体の10%など）
    num_to_modify = max(1, int(1 * num_cols))
    cols_to_modify = np.random.choice(num_cols, size=num_to_modify, replace=False)

    for col_idx in cols_to_modify:
      # noise_width=np.sqrt(1 / (self.cortex_cell[i] + self.num_cell[i]))*0.2*0.998**(self.step_count)
      noise_width=np.sqrt(1 / (self.x2_dim[2] + self.x1_dim[1]))*n_rate
      noise = np.random.uniform(low=-noise_width, high=noise_width , size=matrix[:, col_idx].shape)
      matrix[:, col_idx] += noise
      matrix[:, col_idx] /= np.linalg.norm(matrix[:, col_idx]) + 1e-8
    self.U[4] = matrix

class BP():
  def __init__(self, x_dim, num_batch, lr, f):
      self.lr   = lr
      self.x_dim = x_dim
      self.num_layer  = len(x_dim)-1
      self.num_batch   = num_batch
      self.f = f

      self.initialize_weight() # 重みの初期化

  def get_name(self):
     name = "BP"
     return name

  def initialize_weight(self):
    self.W = [np.random.randn(self.x_dim[i], self.x_dim[i+1]) * np.sqrt(2 / self.x_dim[i])
              for i in range(self.num_layer)]

  def forward(self, X):
    self.xs, self.fxs = [], [X]
    fx = X
    for i in range(self.num_layer):
        x = fx @ self.W[i]
        fx = self.f(x) if i < self.num_layer - 1 else x
        self.xs.append(x)
        self.fxs.append(fx)
    return fx

  def backward(self, X, Y, lr=0.01):
    num_batch = X.shape[0]
    preds = self.fxs[-1]
    error = (preds - Y) / num_batch

    for i in reversed(range(self.num_layer)):
        fx_prev = self.fxs[i]
        dW = fx_prev.T @ error
        if i != 0:
          self.W[i] -= lr * dW

        if i > 0:
          error = (error @ self.W[i].T) * self.f.deriv(self.xs[i - 1])

  def compute_loss(self, Y_true, Y_pred):
    return np.mean((Y_true - Y_pred) ** 2)

  def accuracy(self, Y_true_labels, Y_pred):
    pred_labels = np.argmax(Y_pred, axis=1)
    return np.mean(pred_labels == Y_true_labels)

  def save_cossim(self):
    return 1
  
class FA(BP):

  def get_name(self):
     name = "FA"
     return name

  def initialize_weight(self):
    self.W = [np.random.randn(self.x_dim[i], self.x_dim[i+1]) * np.sqrt(2 / self.x_dim[i])
              for i in range(self.num_layer)]

    self.B = [np.random.randn(self.x_dim[i+1], self.x_dim[i])
              for i in range(self.num_layer)]
    
  def backward(self, X, Y, lr=0.01):
    num_batch = X.shape[0]
    preds = self.fxs[-1]
    error = (preds - Y) / num_batch

    for i in reversed(range(self.num_layer)):
        fx_prev = self.fxs[i]
        dW = fx_prev.T @ error
        if i != 0:
            self.W[i] -= lr * dW

        if i > 0:
            error = (error @ self.B[i]) * self.f.deriv(self.xs[i - 1])

  def save_cossim(self):
    w_vec = self.W[1].flatten()
    b_vec = self.B[1].T.flatten()
    dot = np.dot(w_vec, b_vec)
    norm_w = np.linalg.norm(w_vec) + 1e-10
    norm_b = np.linalg.norm(b_vec) + 1e-10
    cos_theta = dot / (norm_w * norm_b + 1e-10)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return cos_theta