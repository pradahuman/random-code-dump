
import statistics

square_foot = [column[1] for column in data]


home_data = [
    [1, 2100, 3, 350], [1, 1600, 2, 270], [1, 2400, 4, 450], [1, 1410, 3, 232], [1, 1500, 2, 250],
    [1, 2200, 4, 380], [1, 1700, 3, 285], [1, 1800, 3, 310], [1, 1950, 3, 330], [1, 1650, 2, 290],
    [1, 2500, 4, 460], [1, 2800, 4, 510], [1, 3000, 5, 560], [1, 3200, 5, 590], [1, 2000, 3, 340],
    [1, 1550, 3, 250], [1, 2450, 4, 440], [1, 2900, 5, 530], [1, 3150, 4, 570], [1, 1900, 3, 325],
    [1, 1700, 3, 285], [1, 1600, 3, 260], [1, 1400, 2, 230], [1, 1450, 2, 240], [1, 1850, 3, 320],
    [1, 2750, 4, 500], [1, 2250, 3, 390], [1, 2000, 4, 350], [1, 2400, 5, 460], [1, 2200, 4, 400],
    [1, 2650, 4, 480], [1, 2850, 5, 520], [1, 1500, 2, 250], [1, 2900, 5, 540], [1, 1550, 3, 260],
    [1, 1700, 2, 290], [1, 2100, 3, 350], [1, 2300, 4, 420], [1, 2400, 4, 450], [1, 1800, 3, 310],
    [1, 1350, 2, 225], [1, 1250, 2, 210], [1, 2500, 4, 470], [1, 2650, 5, 490], [1, 2900, 4, 530],
    [1, 3000, 5, 550], [1, 3300, 6, 620], [1, 3500, 6, 640], [1, 1600, 3, 260], [1, 1950, 3, 330],
    [1, 2750, 5, 510], [1, 2500, 4, 470], [1, 1700, 3, 280], [1, 1600, 3, 260], [1, 1800, 3, 300],
    [1, 1500, 3, 240], [1, 1550, 2, 250], [1, 2000, 4, 340], [1, 2800, 4, 520], [1, 3100, 5, 570],
    [1, 3400, 6, 620], [1, 3300, 6, 600], [1, 2600, 4, 490], [1, 2200, 3, 380], [1, 2100, 4, 370],
    [1, 1800, 3, 310], [1, 1600, 3, 265], [1, 1400, 2, 230], [1, 1750, 3, 290], [1, 2000, 4, 350],
    [1, 2150, 3, 360], [1, 1950, 3, 335], [1, 1700, 2, 280], [1, 2400, 4, 450], [1, 2600, 5, 490],
    [1, 2750, 4, 510], [1, 2000, 3, 340], [1, 1800, 3, 310], [1, 2500, 4, 460], [1, 3000, 5, 570],
    [1, 3250, 6, 610], [1, 2000, 4, 360], [1, 2800, 5, 530], [1, 1600, 2, 260], [1, 1850, 3, 325],
    [1, 2000, 4, 350], [1, 2300, 4, 410], [1, 1500, 2, 240], [1, 1400, 2, 230], [1, 1900, 3, 330],
    [1, 2200, 4, 390], [1, 2400, 5, 460], [1, 1650, 3, 280], [1, 1700, 3, 285], [1, 1850, 3, 320],
    [1, 1950, 3, 330], [1, 2100, 3, 350], [1, 2350, 4, 430], [1, 2600, 5, 490], [1, 2800, 5, 510],
    [1, 1700, 2, 280], [1, 1600, 2, 260], [1, 1550, 2, 250], [1, 1850, 3, 320], [1, 2100, 4, 370],
    [1, 2300, 4, 420], [1, 2400, 5, 460], [1, 1300, 2, 220], [1, 1400, 2, 230], [1, 1650, 3, 270],
    [1, 1750, 3, 285], [1, 1800, 3, 300], [1, 2000, 3, 340], [1, 2250, 4, 390], [1, 2550, 5, 460],
    [1, 2650, 4, 470], [1, 2750, 5, 510], [1, 1950, 3, 330], [1, 1500, 2, 250], [1, 1700, 3, 280],
    [1, 1750, 3, 290], [1, 1800, 3, 300], [1, 1600, 2, 260], [1, 1450, 2, 240], [1, 2400, 5, 450],
    [1, 2700, 5, 490], [1, 2900, 6, 530], [1, 2200, 4, 400], [1, 1800, 3, 310], [1, 1900, 3, 330],
    [1, 2400, 4, 450], [1, 2650, 5, 490], [1, 2900, 5, 540], [1, 2200, 4, 400], [1, 2100, 3, 350],
    [1, 1500, 2, 240], [1, 1800, 3, 310], [1, 3200, 5, 590], [1, 3400, 6, 620], [1, 3100, 5, 570],
    [1, 2750, 4, 510], [1, 3000, 5, 560], [1, 2400, 5, 460], [1, 2300, 4, 420], [1, 1700, 3, 280],
    [1, 1450, 2, 240], [1, 1800, 3, 305], [1, 2650, 5, 480], [1, 1400, 2, 230], [1, 1300, 2, 220],
    [1, 3100, 5, 570], [1, 1900, 3, 325], [1, 1500, 2, 250], [1, 1600, 3, 260], [1, 2200, 4, 400],
    [1, 1700, 3, 280], [1, 2000, 3, 340], [1, 1850, 3, 325], [1, 3200, 6, 600], [1, 2000, 4, 350],
    [1, 1600, 2, 260], [1, 2100, 4, 360], [1, 1750, 3, 285], [1, 2300, 4, 410], [1, 2700, 5, 490],
    [1, 2500, 4, 460], [1, 2800, 4, 510], [1, 1700, 3, 285], [1, 2200, 4, 400], [1, 2000, 3, 34 ]

]


square_foot = [column[1] for column in home_data]
house_price= [column[3] for column in home_data]
mean_value = statistics.mean(square_foot)




# make function to min/max scale and apply it to data






# create two objects for train and test
class Dataset:

    def __init__(self, data) -> None:
        self.data = data

    def return_entry(self, n):
        return self.data[n]
    
    def return_input(self, n):
        return self.data[n][:3]
    
    def return_inputs(self):
        return [row[:3] for row in self.data]
    
    def return_output(self, n):
        return self.data[n][3]
    
    def return_outputs(self):
        return [row[3] for row in self.data]
    
    def normalize_data(self):


# pass initial random parameters or just like 1,1,1, theta0 isnt trained
class Model:
    def __init__(self, params):
        self.theta0 = params[0]
        self.theta1 = params[1]
        self.theta2 = params[2]

    def return_hx(self, val):
        model_pred = self.theta0*val[0] + self.theta1*val[1] + self.theta2*val[2]
        return model_pred
    
    def return_thetas(self):
        return [self.theta0, self.theta1, self.theta2]
    
    def update_thetas(self, new_thetas):
        self.theta0 = new_thetas[0]
        self.theta1 = new_thetas[1]
        self.theta2 = new_thetas[2]
    
training_set = Dataset(home_data[:200])
test_set = Dataset(home_data[200:])

init_params = [1, 1, 1]

model1 = Model(init_params)

lr = 0.02
train_epochs = 5

'''
batch gradient descent, computes sum of gradient across all training examples and changes
thetas accordingly, once per epoch (converges better but is computationally expensive) (need to average across all training m)

stochastic gradient descent
'''


def training_bgd(model, training_data, learning_rate, epochs):

    for epoch in range(epochs):
        gradient = [0,0,0]
        for m in range(len(training_data.data)):
            unsquared_error = model.return_hx(training_data.return_input(m)) - training_data.return_output(m)
            for i in range(len(gradient)):
                gradient[i] += unsquared_error*training_data.return_input(m)[i]
        
        gradient = list(map(lambda n : n/len(training_data.data), gradient))
    
        gradient = [learning_rate*n for n in gradient]
        updated_thetas = model.return_thetas()

        for j in range(len(model.return_thetas())):
            updated_thetas[j] -= gradient[j]
        model.update_thetas(updated_thetas)


    return model



def training_sgd(model, training_data, learning_rate, epochs):

    for epoch in range(epochs):
        
        for m in range(len(training_data.data)):
            gradient = [0,0,0]
            xm_input = training_data.return_input(m)
            unsquared_error = model.return_hx(xm_input) - training_data.return_output(m)
            for i in range(len(gradient)):
                gradient[i] += unsquared_error*xm_input[i]
        
            gradient = [learning_rate*n for n in gradient]
            updated_thetas = model.return_thetas()

            for j in range(len(updated_thetas)):
                updated_thetas[j] -= gradient[j]
            model.update_thetas(updated_thetas)


    return model
        

                

training_set = Dataset(home_data[:140])
test_set = Dataset(home_data[140:])

init_params = [0, 0, 0]

model_sgd = Model(init_params[:])
model_bgd = Model(init_params[:])

lr = 0.001
train_epochs = 10

training_bgd(model_bgd, training_set, lr, train_epochs)
training_sgd(model_sgd, training_set, lr, train_epochs)





print(test_set.return_input(0))
print(test_set.return_output(0))

pred_out_bgd = model_bgd.return_hx(test_set.return_input(0))
print(pred_out_bgd)




    



    