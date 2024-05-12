import numpy as np
import pandas as pd
from tqdm import tqdm


# Uncomment to use ray if working with large dataset
# import ray
# from ray.util.actor_pool import ActorPool
# @ray.remote
class LRSCalculationWorker:
    """
    Failed caption detection concept by Zippy (github.com/aredden) & ProGamerGov (github.com/progamergov)
    with help from uptightmoose.
    This class utilizes dynamic programming to compute the Longest Repeating Subsequence (LRS)
    of a given string efficiently. Repeating sequences in outputs are suspected to be caused by
    greedy search algorithms.

    Cropping the final 128 characters from a string works the best according to testing, with captions scoring
    above the threshold of 70.0 being likely failures

    See: https://github.com/ProGamerGov/VLM-Captioning-Tools for more examples of usage.
    """

    def __init__(self) -> None:
        # from numba import jit

        # @jit(nopython=True, cache=True) # Remove if not using numba JIT
        def lrs(s1, i, j, dp):
            """
            Compute the Longest Repeating Subsequence (LRS) of a given string.

            Args:
                s1 (str): The input string.
                i (int): Starting index for comparison.
                j (int): Starting index for comparison.
                dp (2D array): Dynamic programming table to store computed results.
            Returns:
                int: The length of the Longest Repeating Subsequence.
            Notes:
                This function uses memoization (dynamic programming) to efficiently compute
                the Longest Repeating Subsequence (LRS) of a given string.
            """

            # return if we have reached the
            # end of either string
            if i >= len(s1) or j >= len(s1):
                return 0

            if dp[i][j] != -1:
                return dp[i][j]

            # while dp[i][j] is not computed earlier
            if dp[i][j] == -1:

                # if characters at index m and n matches
                # and index is different
                # Index should not match
                if s1[i] == s1[j] and i != j:
                    dp[i][j] = 1 + lrs(s1, i + 1, j + 1, dp)

                # else if characters at index m and n don't match
                else:
                    dp[i][j] = max(lrs(s1, i, j + 1, dp), lrs(s1, i + 1, j, dp))

            # return answer
            return dp[i][j]

        self.lrs_fn = lrs

    def find_lrs(self, string):
        """
        Find the Longest Repeating Subsequence (LRS) of a given string.

        Args:
            string (str): The input string.
        Returns:
            tuple: A tuple containing the input string and the length of its Longest Repeating Subsequence.
        Notes:
            This method utilizes the previously initialized lrs_fn to compute the LRS.
        """
        leng = len(string) + 1
        arr = np.zeros((leng, leng))
        arr.fill(-1)
        return string, self.lrs_fn(string[-128:], 0, 0, arr)


# Uncomment to use Ray to speed things up for larger datasets
# workers = [LRSCalculationWorker.remote() for _ in range(8)]
# workers = ActorPool(workers)
# Use this for the for loop if using ray and remove the 'result = lrs_module.find_lrs(caption_string)[1]' line:
# for caption_string, result in tqdm(workers.map_unordered(lambda lrs_module, w: lrs_module.find_lrs.remote(w), caps), total=len(caps))


verbose=True  # Whether or not to print each detected failure
parquet_file_path = 'path/to/parquet_file.parquet'  # Caption loading with differ based on dataset format
df = pd.read_parquet(parquet_file_path)
caps = df["long_caption"].values.tolist()

print("Checking", len(caps), "captions")
count = 0
threshold = 70
lrs_module = LRSCalculationWorker()

for caption_string in tqdm(caps, total=len(caps)):
    result = lrs_module.find_lrs(caption_string)[1]
    if result > threshold or len(caption_string) < 16:
        if verbose:
            print(caption_string[:10], result)
        count+=1
print("Found " + str(count) + " with repeats greater than", threshold)
