def relu(x) :
        return np.maximum(x,0)

def relu_der(x) :
        if x > 0 : 
            return 1
        else : 
            return 0

class neuralnet() :

    def __init__(self,w1,w2,b) :
        self.w1 = np.array(t)
        self.w2 = np.array(t)
        self.b = np.array(t)
        self.lr = 0.01
        self.epochs_to_train = 40

    

    def train(self,x,y) :

        w1 = self.w1
        w2 = self.w2
        b = self.b
        h1 = np.dot(w1,x)+b
        a1 = relu(h1)
        h2 = np.dot(w2,a1)
        a2 = relu(h2)

        dloss_da2 = a2-y
        da2_dh2 = np.vectorize(relu_der)(h2)
        dh2_da1 = w2
        da1_dh1 = np.vectorize(relu_der)(h1)
        dh1_db = 1


        dloss_dw2 = (dloss_da2
                *da2_dh2
                *a1.T
                )
        dloss_dw1 = (dloss_da2
            *da2_dh2
            *dh2_da1
            *da1_dh1
            *x.T)
        dloss_b = dloss_da2*da2_dh2*dh2_da1*da1_dh1*dh1_db

        self.w1 -= lr*dloss_dw1
        self.w2 -= lr*dloss_dw2
        self.b -=  lr*dloss_b

    def predict(self,x):
        w1 = self.w1
        w2 = self.w2
        b = self.b
        h1 = np.dot(x,w1.T)+b
        a1 = relu(h1)
        h2 = np.dot(a1,w2.T)
        a2 = relu(h2)

        return a2.item()

    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            for x,y in train:
                self.train(x, y)
