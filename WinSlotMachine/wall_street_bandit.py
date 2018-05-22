from __future__ import division, print_function
import tensorflow as tf
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import preprocess
import bandit_bot

# Set some overall training parameters
# We will need many more training periods than the toy example.
total_episodes = 100000 # Set total number of episodes to train agent on.
mini_epoch_size = 100
print_epoch_size = 10000

def run():
	# Get our raw data. Totally ripped off the process script from the other Wall Street tutorial, so it's a little inefficient, but it'll work for our purposes
	raw_data = pd.read_csv('./sp500.csv', header=None)
	x_train, y_train, x_test, y_test = preprocess.load_data('./sp500.csv', mini_epoch_size, False)
	print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
	# Sanity check.
	assert mini_epoch_size < 0.5*len(x_train), 'Mini epoch size too large'
	# Have a gander at the stock data. As you can see, the overall trend is upwards
	plt.plot(raw_data[0])
	# Determine how much the simple buy and hold will appreciate over the time frame
	stock_ary = raw_data[0].values
	buy_and_hold = stock_ary[-1] / stock_ary[0]
	print("buy_and_hold", buy_and_hold)
	# Calc average return for basic buy and hold strategy over each mini-epoch period
	avg_return = buy_and_hold** (mini_epoch_size / len(x_train))
	print("avg_return", avg_return)

	# Create our ensemble
	bots = [bandit_bot.TheWimp(),
			bandit_bot.BuyHold(roi=avg_return),
			bandit_bot.TheMonkey(),
			bandit_bot.TheBull(),
			bandit_bot.TheBear(),
			bandit_bot.StratBull(),
			bandit_bot.StratBear()]
	num_bandits = len(bots)

	# init our agent
	tf.reset_default_graph()
	# These two lines established the feed-forward part of the network.
	# This does the actual choosing.
	weights = tf.Variable(tf.ones([num_bandits]))
	chosen_action = tf.argmax(weights,0)
	# The next six lines establish the training proceedure.
	# We feed the reward and chosen action into the network
	# to compute the loss, and use it to update the network.
	reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
	action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
	responsible_weight = tf.slice(weights,action_holder,[1])
	loss = -(tf.log(responsible_weight)*reward_holder)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
	update = optimizer.minimize(loss)

	# Reinforcement training
	sample_ratio  = total_episodes / mini_epoch_size
	print('Mini-epoch size: {}\nNum Mini-epochs: {}'.format(mini_epoch_size, total_episodes // mini_epoch_size))
	total_reward = np.zeros(num_bandits) #Set scoreboard for bandits to 0.
	e = 0.2 #Set the chance of taking a random action.

	init = tf.initialize_all_variables()
	verbose_updates = False
	# Launch the tensorflow graph
	with tf.Session() as sess:
		sess.run(init)
		for i in tqdm(range(total_episodes)):
			# print('Ep {} of {}'.format(i, total_episodes))
			tp = np.random.randint(0, len(x_train))
			x = x_train[tp].ravel() # select time period
			# Choose either a random action or one from our network.
			if np.random.rand(1) < e:
				action = np.random.randint(num_bandits)
			else:
				action = sess.run(chosen_action)
			# reward = pullBandit(bandits[action]) #Get our reward from picking one of the bandits.
			reward = bots[action](x) # Get our reward from picking one of the bandits.
			# Update the network.
			_,resp,ww = sess.run([update,responsible_weight,weights], feed_dict={reward_holder:[reward],action_holder:[action]})
			# Update our running tally of scores.
			total_reward[action] += reward

			if i % print_epoch_size == 0 and verbose_updates:
				print('Results: ', ' '.join(['{:.3f}'.format(bot.p) for bot in bots]))
				print("Running reward: {}".format(str(total_reward)))
			if i % mini_epoch_size == 0:
				[bot.reset() for bot in bots]
	winner = np.argmax(ww)
	print("The agent thinks bandit {} is the most promising....".format(bots[winner].name))

	for i, bot in enumerate(bots):
		star = '*' if i == winner else ''
		print('{: >10}: {: >+10.3f} {}'.format(bot.name, 1000*total_reward[i]/total_episodes, star))

	print('Mini-epoch size: {}\nNum Mini-epochs: {}'.format(mini_epoch_size, total_episodes // mini_epoch_size))
	total_reward = np.zeros(num_bandits) # Set scoreboard for bandits to 0.
	init = tf.initialize_all_variables()
	verbose_updates = False
	# Launch the tensorflow graph
	with tf.Session() as sess:
		sess.run(init)
		for i in tqdm(range(total_episodes)):
			tp = np.random.randint(0, len(x_train))
			x = x_train[tp].ravel() # select time period
			for action in range(len(bots)):
				reward = bots[action](x) # Get our reward from picking one of the bandits.
				# Update our running tally of scores.
				total_reward[action] += reward
			if i % print_epoch_size == 0 and verbose_updates:
				print('Results: ', ' '.join(['{:.3f}'.format(bot.p) for bot in bots]))
				print("Running reward: {}".format(str(total_reward)))
			if i % mini_epoch_size == 0:
				[bot.reset() for bot in bots]
	winner = np.argmax(total_reward)
	print("The agent thinks {} is the most promising....".format(bots[winner].name))

	for i, bot in enumerate(bots):
		star = '*' if i == winner else ''
		print('{: >10}: {: >+10.3f} {}'.format(bot.name, 10*total_reward[i]/total_episodes, star))

	return 0


def main():
	run()
	return 0


if __name__ == '__main__':
	main()
