import numpy as np

class KalmanFilter(object):
    def __init__(self):
        pass
    
class KalmanFilterException(Exception):
    pass

class ParticleFilter(object):
    def __init__(self,sigma:np.ndarray, init_mean:np.ndarray, n_particle:int)->None:
        """
        Args:
            n_particle (int): number of particles
            sigma (np.ndarray): variant-covariant matrix
        """
        self.n_particle = n_particle
        self.sigma = sigma
        self.log_likelihood = -np.inf
        
        self.particles = np.random.multivariate_normal(init_mean, sigma , n_particle)
        self.weights = np.ones((1,n_particle))/n_particle
        self.norm_weights = self.weights

    def norm_likelihood(self, observed_position:np.ndarray, predicted_particle_position:np.ndarray) -> float:
        return float(np.sqrt((2*np.pi)**len(observed_position) * np.linalg.det(self.sigma))**(-1) * np.exp(-1/2 * (observed_position-predicted_particle_position).T @ np.linalg.inv(self.sigma) @ (observed_position-predicted_particle_position)))

    def f_inv(self, w_cumsum, idx, u)->int:
            if np.any(w_cumsum < u) == False:
                return 0
            k = np.max(idx[w_cumsum < u])
            return k+1

    def resampling(self)->None:
        """
        計算量の少ない層化サンプリング
        """
        idx = np.asanyarray(range(self.n_particle))
        u0 = np.random.uniform(0, 1/self.n_particle)
        u = [1/self.n_particle*i + u0 for i in range(self.n_particle)]
        w_cumsum = np.cumsum(self.norm_weights)
        for i, val in enumerate(u):
            self.particles[i] = self.particles[self.f_inv(w_cumsum,idx, val)]
        # self.particles += np.random.multivariate_normal([0,0,0], self.sigma, self.n_particle)
    
    # パーティクルを1ステップ進める
    def step(self)->np.ndarray:
        return self.particles + np.random.multivariate_normal([0,0,0], self.sigma, self.n_particle)

    def estimate(self, observed_position:np.ndarray)->np.ndarray:
        """
        尤度の重みで加重平均した値でフィルタリングされた値を算出
        """
        # パーティクルを1ステップ進める
        self.particles = self.step()
        for i,particle in enumerate(self.particles):
            self.weights[0,i] = self.norm_likelihood(observed_position, np.matrix(particle).T)
        self.norm_weights = self.weights/np.sum(self.weights)
        return self.particles.T @ self.norm_weights.T

    def simulate(self):
        # 時系列データ数
        T = len(self.y)

        # 潜在変数
        x = np.zeros((T+1, self.n_particle))
        x_resampled = np.zeros((T+1, self.n_particle))

        # 潜在変数の初期値
        initial_x = np.random.normal(0, 1, size=self.n_particle)   #--- (1)
        x_resampled[0] = initial_x
        x[0] = initial_x

        # 重み
        w        = np.zeros((T, self.n_particle))
        w_normed = np.zeros((T, self.n_particle))

        l = np.zeros(T) # 時刻毎の尤度

        for t in range(T):
            print("\r calculating... t={}".format(t), end="")
            for i in range(self.n_particle):
                # 1階差分トレンドを適用
                v = np.random.normal(0, np.sqrt(self.alpha_2*self.sigma_2)) # System Noise　#--- (2)
                x[t+1, i] = x_resampled[t, i] + v # システムノイズの付加
                w[t, i] = self.norm_likelihood(self.y[t], x[t+1, i], self.sigma_2) # y[t]に対する各粒子の尤度
            w_normed[t] = w[t]/np.sum(w[t]) # 規格化
            l[t] = np.log(np.sum(w[t])) # 各時刻対数尤度

            # Resampling
            #k = self.resampling(w_normed[t]) # リリサンプリングで取得した粒子の添字
            k = self.resampling2(w_normed[t]) # リリサンプリングで取得した粒子の添字（層化サンプリング）
            x_resampled[t+1] = x[t+1, k]

        # 全体の対数尤度
        self.log_likelihood = np.sum(l) - T*np.log(self.n_particle)

        self.x = x
        self.x_resampled = x_resampled
        self.w = w
        self.w_normed = w_normed
        self.l = l
    
class ParticleFilterException(Exception):
    pass