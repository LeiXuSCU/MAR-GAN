"""
Created by Xu Lei on 2021/09/20 21:30.
E-mail address is leixu@stu.scu.edu.cn.
Copyright  2021 Xu Lei. SCU. All Rights Reserved.
"""
from training.coach import Coach
from utils.config_util import initial_config


def main():
    opts = initial_config()
    coach = Coach(opts)
    coach.train()


if __name__ == '__main__':
    main()
