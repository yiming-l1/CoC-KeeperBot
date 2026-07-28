"""
Unit tests for game mechanics (dice rolls, skill checks, sanity calculations).
"""

import unittest
import random
from src.mechanics import roll_d100, check_skill_success, calculate_sanity_loss


class TestMechanics(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Seed random for reproducibility in some tests
        random.seed(42)

    def test_roll_d100(self):
        """Test that d100 roll returns a number between 1 and 100."""
        for _ in range(100):
            result = roll_d100()
            self.assertGreaterEqual(result, 1)
            self.assertLessEqual(result, 100)

    def test_check_skill_success_critical(self):
        """Test critical success (roll 01)."""
        # Force roll = 1 by mocking, but we'll test the logic
        # Since we can't easily mock, we'll test multiple times
        found_critical = False
        for _ in range(1000):
            roll, success = check_skill_success(50)
            if roll == 1:
                self.assertEqual(success, "Critical Success")
                found_critical = True
                break
        # With 1000 rolls, we should find at least one critical success
        self.assertTrue(found_critical, "Should find critical success in 1000 rolls")

    def test_check_skill_success_fumble_skill_low(self):
        """Test fumble rule: skill < 50, roll >= 96 is fumble."""
        found_fumble = False
        for _ in range(1000):
            roll, success = check_skill_success(40)  # skill < 50
            if roll >= 96:
                self.assertEqual(success, "Fumble")
                found_fumble = True
                break
        self.assertTrue(found_fumble, "Should find fumble in 1000 rolls (skill < 50)")

    def test_check_skill_success_fumble_skill_high(self):
        """Test fumble rule: skill >= 50, only roll 100 is fumble."""
        found_fumble = False
        for _ in range(5000):
            roll, success = check_skill_success(60)  # skill >= 50
            if roll == 100:
                self.assertEqual(success, "Fumble")
                found_fumble = True
            elif roll >= 96 and roll < 100:
                # For skill >= 50, rolls 96-99 should NOT be fumble
                self.assertNotEqual(
                    success,
                    "Fumble",
                    f"Roll {roll} should not be fumble for skill >= 50",
                )
        self.assertTrue(found_fumble, "Should find fumble (roll 100) in 5000 rolls")

    def test_check_skill_success_regular(self):
        """Test regular success/failure."""
        skill_level = 50
        found_success = False
        found_failure = False

        for _ in range(500):
            roll, success = check_skill_success(skill_level)
            if roll <= skill_level and roll > 1:
                # Should be some level of success (not critical, not fumble)
                self.assertIn(
                    success, ["Regular Success", "Hard Success", "Extreme Success"]
                )
                found_success = True
            elif roll > skill_level and roll < 96:
                # Should be failure (but not fumble for skill >= 50)
                if roll != 100:
                    self.assertEqual(success, "Failure")
                    found_failure = True

        self.assertTrue(found_success, "Should find successes")
        self.assertTrue(found_failure, "Should find failures")

    def test_check_skill_success_hard_extreme(self):
        """Test hard and extreme success levels."""
        skill_level = 60
        hard_target = skill_level // 2  # 30
        extreme_target = skill_level // 5  # 12

        for _ in range(1000):
            roll, success = check_skill_success(skill_level)
            if 1 < roll <= extreme_target:
                self.assertEqual(success, "Extreme Success")
            elif extreme_target < roll <= hard_target:
                self.assertEqual(success, "Hard Success")
            elif hard_target < roll <= skill_level:
                self.assertEqual(success, "Regular Success")

    def test_calculate_sanity_loss_fixed(self):
        """Test fixed sanity loss amounts."""
        # Test zero loss
        new_san, loss_val, temp_ins = calculate_sanity_loss(50, "0")
        self.assertEqual(new_san, 50)
        self.assertEqual(loss_val, 0)
        self.assertFalse(temp_ins)

        # Test small fixed loss (< 5, should not trigger temp_insanity)
        new_san, loss_val, temp_ins = calculate_sanity_loss(50, "3")
        self.assertEqual(new_san, 47)
        self.assertEqual(loss_val, 3)
        self.assertFalse(temp_ins)

        # Test large fixed loss (>= 5, should trigger temp_insanity)
        new_san, loss_val, temp_ins = calculate_sanity_loss(50, "5")
        self.assertEqual(new_san, 45)
        self.assertEqual(loss_val, 5)
        self.assertTrue(temp_ins)

        new_san, loss_val, temp_ins = calculate_sanity_loss(50, "10")
        self.assertEqual(new_san, 40)
        self.assertEqual(loss_val, 10)
        self.assertTrue(temp_ins)

    def test_calculate_sanity_loss_dice_format(self):
        """Test dice format sanity loss (e.g., 1d6, 2d4)."""
        current_san = 50

        # Test 1d6 format
        new_san, loss_val, temp_ins = calculate_sanity_loss(current_san, "1d6")
        self.assertGreaterEqual(loss_val, 1)
        self.assertLessEqual(loss_val, 6)
        self.assertEqual(new_san, current_san - loss_val)
        self.assertEqual(temp_ins, loss_val >= 5)

        # Test d6 format (should be treated as 1d6)
        new_san2, loss_val2, temp_ins2 = calculate_sanity_loss(current_san, "d6")
        self.assertGreaterEqual(loss_val2, 1)
        self.assertLessEqual(loss_val2, 6)

        # Test 2d4 format
        new_san, loss_val, temp_ins = calculate_sanity_loss(current_san, "2d4")
        self.assertGreaterEqual(loss_val, 2)
        self.assertLessEqual(loss_val, 8)
        self.assertEqual(new_san, current_san - loss_val)
        self.assertEqual(temp_ins, loss_val >= 5)

    def test_calculate_sanity_loss_bounds(self):
        """Test that sanity cannot go below 0."""
        new_san, loss_val, temp_ins = calculate_sanity_loss(3, "10")
        self.assertEqual(new_san, 0)
        self.assertGreaterEqual(loss_val, 10)
        self.assertTrue(temp_ins)

        new_san, loss_val, temp_ins = calculate_sanity_loss(0, "5")
        self.assertEqual(new_san, 0)

    def test_calculate_sanity_loss_temp_insanity_threshold(self):
        """Test that temp_insanity triggers at exactly 5 SAN loss."""
        # Loss of 4 should NOT trigger
        new_san, loss_val, temp_ins = calculate_sanity_loss(50, "4")
        self.assertFalse(temp_ins, "Loss of 4 should NOT trigger temp_insanity")

        # Loss of 5 SHOULD trigger
        new_san, loss_val, temp_ins = calculate_sanity_loss(50, "5")
        self.assertTrue(temp_ins, "Loss of 5 SHOULD trigger temp_insanity")

        # Loss of 6 SHOULD trigger
        new_san, loss_val, temp_ins = calculate_sanity_loss(50, "6")
        self.assertTrue(temp_ins, "Loss of 6 SHOULD trigger temp_insanity")


if __name__ == "__main__":
    unittest.main()
