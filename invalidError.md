```py
import onnx_tool
import numpy
import json

# Variable to store JSON data for profiler which we will merge in finalCopy used for Partition and Scheduling
profilerDataForEachBlockInJson = []

def process_model(modelpath, input_shape):
    profiler_data_path = modelpath + "ProfilerData"
    m = onnx_tool.Model(modelpath)
    m.graph.shape_infer({'data': numpy.zeros(input_shape)})  # Update tensor shapes with new input tensor
    m.graph.profile()
    m.graph.print_node_map(profiler_data_path)  # Save file

    # Initialize variables to store the extracted data
    output_shape = None
    forward_MACs = None
    memory_requirement = None

    # Read the file and extract the required information
    with open(profiler_data_path, 'r') as file:
        lines = file.readlines()  # Read all lines in the file

        # Extract the second last line and get the last column value (Output Shape)
        second_last_line = lines[-2]  # Get the second last line
        output_shape = second_last_line.split()[-1]  # Split the line and get the last element

        # Extract the last line and get the Forward_MACs value
        last_line = lines[-1]  # Get the last line
        forward_MACs = last_line.split()[2]  # Split the line and get the third element (Forward_MACs)
        memory_requirement = last_line.split()[4]  # Split the line and get the fifth element (Memory Requirement)

    # Print the extracted values
    print("ModelPath:", {modelpath})
    print("OutputShape:", output_shape)
    print("ForwardMACs:", forward_MACs)
    print("MemoryRequirement:", memory_requirement)

    # Create a dictionary with the data
    data_dict = {
        "ModelPath": modelpath,
        "OutputShape": output_shape,
        "ForwardMACs": forward_MACs,
        "MemoryRequirement": memory_requirement
    }

    # Convert "Forward MACs" and "Memory Requirement" to integers
    data_dict["ForwardMACs"] = int(data_dict["ForwardMACs"].replace(",", ""))
    data_dict["MemoryRequirement"] = int(data_dict["MemoryRequirement"].replace(",", ""))

    # Append the dictionary to the list
    profilerDataForEachBlockInJson.append(data_dict)

    # Convert output_shape string to tuple for the next iteration
    next_input_shape = tuple(map(int, output_shape.replace('x', ' ').split()))
    return next_input_shape

# Initial input shape
current_input_shape = (1, 1, 1, 128)


# Process each model
for path in output_paths:
    current_input_shape = process_model(path, current_input_shape)

# profilerDataForEachBlockInJson Store in JSON file
# After the loop, write the list of dictionaries to a JSON file
with open('model_data.json', 'w') as file:
    json.dump(profilerDataForEachBlockInJson, file, indent=4)

# Merge Data and get final result
Data3 = []



# Create a mapping from ModelPath to dictionary in Data2 for quick lookup
model_path_to_data2 = {item['ModelPath']: item for item in profilerDataForEachBlockInJson}

# Iterate over each item in Data1 and merge with corresponding item in Data2
for item1 in model_parts:
    model_path = item1['output_path']
    if model_path in model_path_to_data2:
        # Merge the two dictionaries
        merged_data = {**item1, **model_path_to_data2[model_path]}
        Data3.append(merged_data)

# Print or return Data3
print(Data3)

with open('FinalWorkI_needForPartitionigAndScheduling.json', 'w') as file:
    json.dump(Data3, file, indent=4)
```

```bash
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[8], line 64
     62 # Process each model
     63 for path in output_paths:
---> 64     current_input_shape = process_model(path, current_input_shape)
     66 # profilerDataForEachBlockInJson Store in JSON file
     67 # After the loop, write the list of dictionaries to a JSON file
     68 with open('model_data.json', 'w') as file:

Cell In[8], line 11, in process_model(modelpath, input_shape)
      9 profiler_data_path = modelpath + "ProfilerData"
     10 m = onnx_tool.Model(modelpath)
---> 11 m.graph.shape_infer({'data': numpy.zeros(input_shape)})  # Update tensor shapes with new input tensor
     12 m.graph.profile()
     13 m.graph.print_node_map(profiler_data_path)  # Save file

File ~/miniconda/envs/llmexperiment_p38/lib/python3.8/site-packages/onnx_tool/graph.py:974, in Graph.shape_infer(self, inputs)
    972 in_valid, tname = self.check_inputs()
    973 if not in_valid:
--> 974     raise ValueError(
    975         f"The input tensor {tname}'s shape {self.tensormap[tname].shape2str()} is not valid, Please set it to a valid shape.")
    976 self.shapeinfer_optime_map = {}
    977 from .utils import timer

ValueError: The input tensor input_ids's shape [batch_size,sequence_length] is not valid, Please set it to a valid shape.

```
