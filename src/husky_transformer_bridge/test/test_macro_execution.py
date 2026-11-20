import unittest
from husky_transformer_bridge.macro_execution import MacroExecution


class MacroTests(unittest.TestCase):
    def test_turns_count_and_reward_is_discounted(self):
        cells = {(1, 1), (2, 1), (3, 1)}
        macro = MacroExecution(lambda s, g: [(3, 1), (2, 1), (1, 1)], cells, [7, 2, 2, 2])
        macro.begin(0, (3, 1), 0)
        actions = []
        for _ in range(5):
            p = macro.next_primitive()
            actions.append(p['primitive_action'])
            macro.confirm(p['end'], p['direction'])
        self.assertEqual(actions, [1, 1, 2, 2, 2])
        result = macro.feedback()
        self.assertTrue(result['trace'][-1]['blocked'])
        self.assertAlmostEqual(result['reward'], 7 * .99 ** 4)
        self.assertEqual(result['astar_events'], 1)

    def test_no_commit_before_observed_completion(self):
        macro = MacroExecution(lambda s, g: [s], {(1, 1)}, [1]*4)
        macro.begin(0, (1, 1), 0)
        macro.next_primitive()
        with self.assertRaises(ValueError):
            macro.confirm((2, 1), 0)
        with self.assertRaises(RuntimeError):
            macro.feedback()
        self.assertEqual(macro.episode_steps, 0)

    def test_episode_limit_and_replanning_discount(self):
        macro = MacroExecution(lambda s, g: [s], {(1, 1)}, [7, 2, 2, 2], max_steps=10)
        rewards = []
        for _ in range(2):
            macro.begin(0, (1, 1), 0)
            for _ in range(5):
                p = macro.next_primitive()
                macro.confirm(p['end'], p['direction'])
            rewards.append(macro.feedback()['reward'])
        self.assertAlmostEqual(rewards[0], 7 * .99**4)
        self.assertAlmostEqual(rewards[1], 7 * .99**5)
        self.assertTrue(macro.feedback()['truncated'])
        with self.assertRaises(RuntimeError):
            macro.begin(0, (1, 1), 0)


if __name__ == '__main__':
    unittest.main()
