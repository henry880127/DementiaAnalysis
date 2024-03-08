import pickle

# Specify the path to the pickle file
pickle_file_path = '../Dataset/Dementia_paper_dataset_data.pkl'

# Open the pickle file in read mode
with open(pickle_file_path, 'rb') as file:
    # Load the data from the pickle file
    data = pickle.load(file)

# Now you can use the 'data' variable to access the imported data

print(data.keys())
print(data['train'].shape)
print(data['test'].shape)