import unittest

import chess
import torch
from src.engine.evaluators.deep_cnn_eval import (
    DeepChessCNN,
    DeepCNN_Eval,
    LightweightDeepCNN,
)


class TestDeepCNNEval(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board()

    def test_deep_architecture_creation(self):
        """Test creating deep CNN evaluator with deep architecture."""
        evaluator = DeepCNN_Eval(
            self.board,
            architecture="deep",
            num_residual_blocks=4,  # Smaller for testing
            base_channels=128,
        )
        self.assertIsInstance(evaluator.model, DeepChessCNN)

    def test_lightweight_architecture_creation(self):
        """Test creating deep CNN evaluator with lightweight architecture."""
        evaluator = DeepCNN_Eval(
            self.board,
            architecture="lightweight",
            num_residual_blocks=2,
            base_channels=64,
        )
        self.assertIsInstance(evaluator.model, LightweightDeepCNN)

    def test_evaluation(self):
        """Test that evaluation returns a valid score."""
        evaluator = DeepCNN_Eval(
            self.board,
            architecture="lightweight",
            num_residual_blocks=2,
            base_channels=64,
        )
        score = evaluator.evaluate()
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, -evaluator.MAX_EVAL)
        self.assertLessEqual(score, evaluator.MAX_EVAL)

    def test_checkmate_evaluation(self):
        """Test evaluation of checkmate positions."""
        # Scholar's mate
        board = chess.Board()
        moves = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
        for move in moves:
            board.push(chess.Move.from_uci(move))

        evaluator = DeepCNN_Eval(
            board, architecture="lightweight", num_residual_blocks=1, base_channels=32
        )
        score = evaluator.evaluate()
        # Should return max eval for white (checkmate)
        self.assertEqual(score, evaluator.MAX_EVAL)

    def test_model_info(self):
        """Test model information retrieval."""
        evaluator = DeepCNN_Eval(
            self.board, architecture="deep", num_residual_blocks=2, base_channels=64
        )
        info = evaluator.get_model_info()

        self.assertIn("architecture", info)
        self.assertIn("total_parameters", info)
        self.assertIn("trainable_parameters", info)
        self.assertIn("device", info)
        self.assertIn("model_class", info)

        self.assertEqual(info["architecture"], "deep")
        self.assertGreater(info["total_parameters"], 0)

    def test_different_configurations(self):
        """Test different model configurations."""
        configs = [
            {"architecture": "deep", "num_residual_blocks": 2, "base_channels": 64},
            {
                "architecture": "lightweight",
                "num_residual_blocks": 1,
                "base_channels": 32,
            },
            {"architecture": "deep", "num_residual_blocks": 4, "base_channels": 128},
        ]

        for config in configs:
            with self.subTest(config=config):
                evaluator = DeepCNN_Eval(self.board, **config)
                score = evaluator.evaluate()
                self.assertIsInstance(score, float)

                info = evaluator.get_model_info()
                self.assertEqual(info["architecture"], config["architecture"])


if __name__ == "__main__":
    unittest.main()
