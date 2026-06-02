import os

import torch
import nvflare.client as flare
import argparse
import task

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(description="NVFlare Client Training Script")
    parser.add_argument("--batch_size", type=int, default=32, help="Input batch size for training")
    parser.add_argument("--num_of_clients", type=int, default=32, help="Input batch size for training")
    parser.add_argument("--epochs", type=int, default=32, help="Input batch size for training")
    parser.add_argument("--num_of_rounds", type=int, default=2, help="Input batch size for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    print(f"Successfully loaded arguments!")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Clients: {args.num_of_clients}")
    print(f"Epochs: {args.epochs}")
    print(f"Number of Rounds: {args.num_of_rounds}")
    print(f"Seed: {args.seed}")
    num_partitions = args.num_of_clients
    batch_size = args.batch_size
    epochs = args.epochs
    num_rounds = args.num_of_rounds
    print("Starting client...")
    task.set_seed(args.seed)
    model = task.load_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    client_num = int(client_name.split("-")[1]) - 1
    print("Client name:", client_name)
    print("Client number:", client_num)

    train_loader, test_loader =  task.load_data(num_partitions ,client_num, batch_size)

    while flare.is_running():
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")
        data = {}
        if input_model.params is not None and len(input_model.params) > 0:
            model.load_state_dict(input_model.params)
            _, accuracy = task.test_func(model, test_loader, device)
            data = {"accuracy": accuracy, "instance_count": len(test_loader.dataset), "round": input_model.current_round, "final_round": input_model.current_round == num_rounds}
            print("Global model loaded successfully.")
        else:
            if (input_model.current_round > 0):
                print("Warning: No global weights received, round is greater than 0")
                os.exit(1)
            print("Round 0: No global weights received. Using locally initialized weights.")    
        if num_rounds != input_model.current_round:
            steps = epochs * len(train_loader)
            task.train_func(model, train_loader, epochs, device)
            print("Finished Training")
        else:
            print("Final round, skipping training and sending final model for evaluation.")
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            meta={"NUM_STEPS_CURRENT_ROUND": steps, "data": data}
        )

        flare.send(output_model)


if __name__ == "__main__":
    main()
