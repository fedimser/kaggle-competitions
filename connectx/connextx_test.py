from kaggle_environments import evaluate, make, utils
from connectx import the_agent

def test_agent_against_itself():
    env = make("connectx", debug=True)
    env.run([the_agent, the_agent])
    assert env.state[0].status == env.state[1].status == "DONE"

def test_agent_wins_against_random():
    assert evaluate("connectx", [the_agent, "random"], num_episodes=1)[0][0]==1
    assert evaluate("connectx", ["random", the_agent], num_episodes=1)[0][0]==-1

def test_agent_wins_against_negamax():
    assert evaluate("connectx", [the_agent, "negamax"], num_episodes=1)[0][0]==1
    assert evaluate("connectx", ["negamax", the_agent], num_episodes=1)[0][0]==-1
