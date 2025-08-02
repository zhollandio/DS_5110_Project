from torch.utils.data import TensorDataset
import torch

#add the data path
load_data_path = 'Full_Inference.pt'
#add the path you want to store chunks
save_data_path = ''
# Load the data
data = torch.load(load_data_path)

# Define the chunk size and data length
data_length = len(data.tensors[0])  # Assuming it's a TensorDataset
#set your desired chunk size
chunk_size = 70000
num_chunks = data_length // chunk_size

# Save each chunk separately
for i in range(num_chunks + 1):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, data_length)
    
    # Extracting the chunks correctly from each tensor in the dataset
    chunk_0 = data.tensors[0][start:end].numpy()
    chunk_1 = data.tensors[1][start:end].numpy()
    chunk_2 = data.tensors[2][start:end].numpy()
    
    # Saving the chunks to separate files
    torch.save(TensorDataset(torch.tensor(chunk_0), torch.tensor(chunk_1), torch.tensor(chunk_2)), save_data_path + 'chunk_' + str(i) + '.pt')