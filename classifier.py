import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


print("Importing datasets...")



transformation = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,),(0.5,)),])

training_dataset = torchvision.datasets.MNIST("./", transform=transformation, download=False, train=True)
test_dataset = torchvision.datasets.MNIST("./", transform=transformation, download=False, train=False)

training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=128, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

# shape of training data
dataiter = iter(training_dataloader)
images, labels = dataiter.next()

# visualizing the training images
#plt.imshow(images[0].numpy().squeeze(), cmap='gray')
#plt.show()

print("Done")
print("Constructing network...")

input = 784
layers = [196, 49]
output = 10

network = nn.Sequential(nn.Linear(input,layers[0]),
                        nn.ReLU(),
                        nn.Linear(layers[0], layers[1]),
                        nn.ReLU(),
                        nn.Linear(layers[1], output),
                        nn.LogSoftmax(dim=1))

#Loss function
criterion = nn.NLLLoss()

#Optimiser

opt = optim.SGD(network.parameters(),lr= 0.05)
print("Done")


print("Training the network")
epochs =10
for i in range(epochs):
    running_loss = 0

    # Loop through examples in the set
    for images, labels in training_dataloader:
        images = images.view(images.shape[0], -1)
        #training pass
        opt.zero_grad()

        #network output
        output = network(images)

        #calculate loss
        loss = criterion(output,labels)

        # Perform backpropagation step
        loss.backward()

        # optimizes its weights here
        opt.step()
        running_loss += loss.item()
print("Done")
print("Performing Validation...")

images,labels = next(iter(test_dataloader))
correct_count =0
all_count = 0
for images, labels in test_dataloader:
    for i in range(len(labels)):
        transform_image = images[i].view(1,784)
        with torch.no_grad():
            logps = network(transform_image)

        #num value from output tensor
        ps = torch.exp(logps)
        probability = list(ps.numpy()[0])
        predict_label = probability.index(max(probability))
        true_label = labels.numpy()[i]

        if true_label == predict_label:
            correct_count +=1

        all_count +=1
        print(f'accuracy: {round(correct_count/all_count, 3)}')
print("Done")

print("Number Of Images Tested =", all_count)
#print("Loss", running_loss/all_count)
print("Model Accuracy =", (correct_count/all_count))
print("Done!")


path = input("Please enter a filepath: ")

while path != "exit":
    image = open(path)
    trans_image = np.reshape(image,784)
    image_array = np.array([trans_image])
    image_tensor = torch.from_numpy(image_array.astype(float))
    image_tensor = image_tensor.type(torch.FloatTensor)

    with torch.no_grad():
        logps = network(image_tensor)
    # Get number value from output tensor
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))

    print(f"Classifier: {predict_label}")
    path = input("Please enter a filepath: ")









