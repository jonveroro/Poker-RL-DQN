from pypokerengine.api.game import setup_config, start_poker
from FishPlayer import DataBloggerBot
from Player import RLPlayer
import Brain
import time

while True:

	config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)
	config.register_player(name="Agent", algorithm=RLPlayer())
	config.register_player(name="monte-carlo1", algorithm=DataBloggerBot())
	config.register_player(name="monte-carlo2", algorithm=DataBloggerBot())
	config.register_player(name="monte-carlo3", algorithm=DataBloggerBot())
	config.register_player(name="monte-carlo4", algorithm=DataBloggerBot())
	config.register_player(name="fish1", algorithm=DataBloggerBot())
	config.register_player(name="fish2", algorithm=DataBloggerBot())
	config.register_player(name="fish3", algorithm=DataBloggerBot())
	config.register_player(name="fish4", algorithm=DataBloggerBot())
	config.register_player(name="fish5", algorithm=DataBloggerBot())
	game_result = start_poker(config, verbose=1)

	players = game_result['players']

	agent = players[0]

	stack = agent['stack']

	file = open('logs/Results.csv','a+')
	file.write(str(stack)+'\n')



	#time.sleep(5)