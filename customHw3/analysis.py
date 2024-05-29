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
    answerNoise = 0
    return answerDiscount, answerNoise


def question3a():
    """
    answerDiscount (할인율):
    answerDiscount는 미래의 보상이 현재 보상에 비해 얼마나 가치가 있는지를 나타내는 할인율입니다.
    이 값이 0에 가까울수록 에이전트는 현재의 보상을 더 중요하게 여기고, 1에 가까울수록 미래의 보상도 현재만큼 중요하게 고려합니다.
    여기서 만일 0.1의 값은 에이전트가 미래의 보상을 현재의 보상보다 상대적으로 낮게 평가함을 의미하며,
    짧은 시간 내에 빠른 보상을 추구하는 경향을 나타냅니다.
    answerNoise (노이즈):
    answerNoise는 에이전트의 행동이 얼마나 확실하게 실행될지를 나타내는 매개변수입니다.
    값이 0이면 에이전트가 선택한 행동이 정확히 실행된다는 것을 의미하고, 값이 클수록 행동 결과에 불확실성이 커집니다.
    여기서 0.0은 에이전트의 모든 결정이 정확히 예측대로 실행됨을 보여줍니다.
    answerLivingReward (살아있는 보상):
    answerLivingReward는 각 시간 단계에서 에이전트가 살아있을 때 받는 보상을 나타냅니다.
    이 값이 양수이면 에이전트가 살아있는 것 자체에 보상을 받는 것이고, 0이면 살아있는 동안 추가 보상이 없음을 의미합니다.
    여기서 0.0은 에이전트가 단순히 살아있는 것만으로는 추가 보상을 받지 않는다는 것을 의미합니다.
    """
    answerDiscount = 0.1
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    answerDiscount = 0.2
    answerNoise = 0.1
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    answerDiscount = 0.7
    answerNoise = 0.1
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    answerDiscount = 0.9
    answerNoise = 0.4
    answerLivingReward = 0.6
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    answerDiscount = 0
    answerNoise = 0
    answerLivingReward = 1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
