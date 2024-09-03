import sys
import subprocess

#archs = [(2,5),(2,50),(2,100),(2,200),\
#        (3,5),(3,50),(3,100),(3,200),\
#        (4,5),(4,50),(4,100),(4,200)]
archs = [(2,5),(2,50),(2,100),(2,200)]

#frac_of_data = [i / 10 for i in range(1,11)]
frac_of_data = [0.6,0.7,0.8,0.9,1.0]

model_params_copy = "./model_params_copy.yml"

arch_no = 0
for frac in frac_of_data:
    for arch in archs:

        arch_no += 1
        print(f"Training Arch No. {arch_no}")    
        print(f"Running {arch[0]} layers and {arch[1]} nodes with {frac*100}% of data")
        command = (f"CUDA_VISIBLE_DEVICES={0} python train.py --yaml_file {model_params_copy}"  
                 f" --num_layers {arch[0]} --num_nodes {arch[1]} --frac_of_data {frac}"
                )

        res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        last_lines = res.stdout.splitlines()[-8:]
        decoded_data = [item.decode('utf-8') for item in last_lines]

        formatted_output = "\n".join(decoded_data)

        print(formatted_output)

