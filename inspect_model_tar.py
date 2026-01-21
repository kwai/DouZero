import argparse
import torch


def summarize_state_dict(state_dict):
    for name, tensor in state_dict.items():
        shape = tuple(tensor.shape)
        dtype = str(tensor.dtype)
        print(f"{name}  shape={shape}  dtype={dtype}")


def main():
    parser = argparse.ArgumentParser(description="Inspect DouZero model.tar")
    parser.add_argument("--path", type=str, required=True, help="Path to model.tar")
    args = parser.parse_args()

    ckpt = torch.load(args.path, map_location="cpu")
    print("Top-level keys:", list(ckpt.keys()))
    print("frames:", ckpt.get("frames"))
    print("position_frames:", ckpt.get("position_frames"))
    print("flags:", ckpt.get("flags"))

    model_state = ckpt.get("model_state_dict", {})
    print("model_state_dict positions:", list(model_state.keys()))
    for position in ["landlord", "landlord_up", "landlord_down"]:
        if position in model_state:
            print(f"\n[{position}]")
            summarize_state_dict(model_state[position])


if __name__ == "__main__":
    main()
