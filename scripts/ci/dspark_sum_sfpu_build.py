"""Cached simulator library for the isolated SFPU sum-update candidate."""

from dspark_score_sfpu_build import main
from dspark_sum_sfpu import sum_scope


if __name__ == '__main__':
    with sum_scope():
        main()
