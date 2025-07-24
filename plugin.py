import asyncio
import re
from typing import List

from swift.plugin import ORM, orms


class CosineReward(ORM):
    def __init__(self,
                 tokenizer=None,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 512,
                 soft_cache_length: int = 256):
        self.tokenizer = tokenizer
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.soft_cache_length = soft_cache_length
        self.accuracy_orm = CLSAccuracyORM_choice()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        rewards = []
        for content, acc_reward in zip(completions, acc_rewards):
            # is_correct = acc_reward >= 1
            if acc_reward == 1.0 or acc_reward == 0.0:
                if acc_reward == 1.0:
                    # Swap min/max for correct answers
                    min_value = self.max_len_value_correct
                    max_value = self.min_len_value_correct
                else:
                    min_value = self.max_len_value_wrong
                    max_value = self.min_len_value_wrong
                gen_len = len(self.tokenizer.encode(content))
                gen_len = max(0, gen_len - self.soft_cache_length)   #长度小于256 不做奖励
                max_len = max(0, self.max_len - self.soft_cache_length)
                reward = self.cosfn(gen_len, max_len, min_value, max_value)
                # reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            elif acc_reward == -1.0:
                reward = -1.0
            else:
                raise ValueError("Invalid acc reward")
            rewards.append(reward)
        return rewards


class CLSAccuracyORM_choice(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        # print(completions)
        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0

            content_match = re.search(r'<answer>(.*?)</answer>', content)
            student_answer = content_match.group(1).strip() if content_match else content.strip()

            ground_truth = sol.strip().replace(' ','').replace('_','').replace('.','').replace('\n','').lower()
            student_answer = student_answer.strip().replace(' ','').replace('_','').replace('.','').replace('\n','').lower()

            # Compare the extracted answers
            if ground_truth == student_answer:
                reward = 1.0

            if student_answer not in ['a', 'b', 'c' ,'d']:
                reward = -1.0

            rewards.append(reward)

        return rewards


orms['external_cls_acc_choice'] = CLSAccuracyORM_choice
orms['external_cosine'] = CosineReward