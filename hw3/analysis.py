# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0.0
    return answerDiscount, answerNoise

def question3a():
    # 절벽의 위험을 감수하면서 가까운 출구를 선호하는 유형
    answerDiscount = 0.2
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    # 절벽(-10)을  피하는  동시에  가까운  출구(+1)를  선호하는  유형
    answerDiscount = 0.3
    answerNoise = 0.1
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    # 절벽(-10)의  위험을  감수하면서  먼  출구(+10)를  선호하는  유형
    answerDiscount = 0.8
    answerNoise = 0.0
    answerLivingReward = 0.4
    return answerDiscount, answerNoise, answerLivingReward


def question3d():
    # 절벽(-10)을  피하는  동시에  먼  출구(+10)를  선호하는  유형
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.2
    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    # 출구와  절벽을  모두  피하는  유형(즉,  에피소드가  종료되지  않아야  함)
    answerDiscount = 1.0
    answerNoise = 0.0
    answerLivingReward = 1.0
    return answerDiscount, answerNoise, answerLivingReward

def question8():
    return "NOT POSSIBLE"

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
