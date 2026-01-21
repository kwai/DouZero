"""
Benchmark script for DouZero model validation and inference latency measurement.
"""
import os
import time
import argparse
import pickle
import torch
import numpy as np

from douzero.evaluation.deep_agent import DeepAgent, _load_model
from douzero.env.env import get_obs
from douzero.env.game import GameEnv


def benchmark_model_loading(model_path, position='landlord'):
    """Test if the model can be loaded successfully."""
    print(f"\n{'='*60}")
    print(f"TESTING MODEL LOADING")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Position: {position}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    try:
        start_time = time.time()
        model = _load_model(position, model_path)
        load_time = time.time() - start_time

        print(f"\n[SUCCESS] Model loaded successfully in {load_time:.4f} seconds")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (fp32)")

        return model, True
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, False


def benchmark_inference_latency(model_path, position='landlord', num_iterations=100):
    """Measure inference latency with synthetic data."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING INFERENCE LATENCY")
    print(f"{'='*60}")
    print(f"Number of iterations: {num_iterations}")

    try:
        # Load model
        agent = DeepAgent(position, model_path)

        # Create synthetic observation data
        # Based on the model architecture:
        # - z_batch shape: (num_actions, seq_len, 162) for LSTM input
        # - x_batch shape: (num_actions, 373) for landlord or (num_actions, 484) for farmer

        if position == 'landlord':
            x_dim = 373
        else:
            x_dim = 484

        # Simulate different action counts to test variable action spaces
        action_counts = [5, 10, 20, 50]

        for num_actions in action_counts:
            print(f"\n--- Testing with {num_actions} legal actions ---")

            # Generate synthetic data
            z_batch = np.random.randn(num_actions, 15, 162).astype(np.float32)
            x_batch = np.random.randn(num_actions, x_dim).astype(np.float32)

            obs = {
                'z_batch': z_batch,
                'x_batch': x_batch
            }

            # Warm-up
            z = torch.from_numpy(obs['z_batch']).float()
            x = torch.from_numpy(obs['x_batch']).float()
            if torch.cuda.is_available():
                z, x = z.cuda(), x.cuda()
            with torch.no_grad():
                _ = agent.model.forward(z, x, return_value=True)

            # Benchmark
            latencies = []
            for _ in range(num_iterations):
                z = torch.from_numpy(obs['z_batch']).float()
                x = torch.from_numpy(obs['x_batch']).float()

                start_time = time.time()

                if torch.cuda.is_available():
                    z, x = z.cuda(), x.cuda()

                with torch.no_grad():
                    y_pred = agent.model.forward(z, x, return_value=True)['values']
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _ = y_pred.detach().cpu().numpy()

                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies.append(latency)

            latencies = np.array(latencies)
            print(f"  Mean latency: {np.mean(latencies):.3f} ms")
            print(f"  Median latency: {np.median(latencies):.3f} ms")
            print(f"  Min latency: {np.min(latencies):.3f} ms")
            print(f"  Max latency: {np.max(latencies):.3f} ms")
            print(f"  Std dev: {np.std(latencies):.3f} ms")
            print(f"  P95 latency: {np.percentile(latencies, 95):.3f} ms")
            print(f"  P99 latency: {np.percentile(latencies, 99):.3f} ms")

        return True

    except Exception as e:
        print(f"\n[ERROR] Inference benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_simple_evaluation(landlord_path, eval_data_path, num_games=100):
    """Run a simple evaluation to verify model works in game environment."""
    print(f"\n{'='*60}")
    print(f"RUNNING SIMPLE GAME EVALUATION")
    print(f"{'='*60}")
    print(f"Evaluation data: {eval_data_path}")
    print(f"Number of games: {num_games}")

    try:
        # Load evaluation data
        with open(eval_data_path, 'rb') as f:
            card_play_data_list = pickle.load(f)

        # Limit to specified number of games
        card_play_data_list = card_play_data_list[:num_games]

        # Load players
        from douzero.evaluation.random_agent import RandomAgent

        players = {
            'landlord': DeepAgent('landlord', landlord_path),
            'landlord_up': RandomAgent(),
            'landlord_down': RandomAgent()
        }

        print(f"Setup: DouZero (Landlord) vs Random (Farmers)")

        # Create game environment
        env = GameEnv(players)

        # Run games and measure inference time
        inference_times = []
        game_start_time = time.time()

        for idx, card_play_data in enumerate(card_play_data_list):
            env.card_play_init(card_play_data)
            while not env.game_over:
                # Measure time for each step (includes model inference)
                step_start = time.time()
                env.step()
                step_time = (time.time() - step_start) * 1000
                inference_times.append(step_time)
            env.reset()

            if (idx + 1) % 20 == 0:
                print(f"  Completed {idx + 1}/{num_games} games...")

        total_time = time.time() - game_start_time

        # Results
        num_total_wins = env.num_wins['landlord'] + env.num_wins['farmer']
        landlord_wp = env.num_wins['landlord'] / num_total_wins if num_total_wins > 0 else 0
        farmer_wp = env.num_wins['farmer'] / num_total_wins if num_total_wins > 0 else 0

        landlord_adp = env.num_scores['landlord'] / num_total_wins if num_total_wins > 0 else 0
        farmer_adp = 2 * env.num_scores['farmer'] / num_total_wins if num_total_wins > 0 else 0

        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Games played: {num_games}")
        print(f"Total time: {total_time:.2f} seconds ({total_time/num_games:.2f} sec/game)")
        print(f"\nWinning Percentage:")
        print(f"  Landlord (DouZero): {landlord_wp:.4f} ({landlord_wp*100:.2f}%)")
        print(f"  Farmers (Random):   {farmer_wp:.4f} ({farmer_wp*100:.2f}%)")
        print(f"\nAverage Difference Points:")
        print(f"  Landlord (DouZero): {landlord_adp:.4f}")
        print(f"  Farmers (Random):   {farmer_adp:.4f}")

        print(f"\nInference Statistics:")
        inference_times = np.array(inference_times)
        print(f"  Mean step time: {np.mean(inference_times):.3f} ms")
        print(f"  Median step time: {np.median(inference_times):.3f} ms")
        print(f"  Total steps: {len(inference_times)}")

        return True

    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='DouZero Model Benchmark')
    parser.add_argument('--model_path', type=str,
                       default=r'C:\Users\KyuEu\PycharmProjects\DouZero\baselines\douzero_ADP\landlord.ckpt',
                       help='Path to the model checkpoint')
    parser.add_argument('--position', type=str, default='landlord',
                       choices=['landlord', 'landlord_up', 'landlord_down'],
                       help='Model position')
    parser.add_argument('--eval_data', type=str, default='eval_data.pkl',
                       help='Path to evaluation data')
    parser.add_argument('--num_iterations', type=int, default=100,
                       help='Number of iterations for latency benchmark')
    parser.add_argument('--num_games', type=int, default=100,
                       help='Number of games for evaluation')
    parser.add_argument('--skip_loading', action='store_true',
                       help='Skip model loading test')
    parser.add_argument('--skip_latency', action='store_true',
                       help='Skip inference latency benchmark')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Skip game evaluation')

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# DouZero Model Validation and Benchmark")
    print(f"{'#'*60}")

    # Test 1: Model Loading
    if not args.skip_loading:
        model, success = benchmark_model_loading(args.model_path, args.position)
        if not success:
            print("\n[WARNING] Model loading failed. Stopping benchmark.")
            return

    # Test 2: Inference Latency
    if not args.skip_latency:
        success = benchmark_inference_latency(args.model_path, args.position, args.num_iterations)
        if not success:
            print("\n[WARNING] Inference benchmark failed.")

    # Test 3: Game Evaluation
    if not args.skip_evaluation:
        if not os.path.exists(args.eval_data):
            print(f"\n[WARNING] Evaluation data not found at {args.eval_data}")
            print("Run: python generate_eval_data.py --num_games 1000")
        else:
            success = run_simple_evaluation(args.model_path, args.eval_data, args.num_games)

    print(f"\n{'#'*60}")
    print(f"# Benchmark Complete")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()
