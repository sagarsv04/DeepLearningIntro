import numpy as np, pandas as pd

class  BanditBot :
    name = 'BaseBot'
    inflation_loss = 0.995 # Due to inflation, holding on to cash is equivalent to losing money. We coarsely simulate this with this parameter
    def __init__(self, roi=1., bankroll=1., nt=50, p=1., q=0., r_buy=1., r_sell=1.):
        self.roi = roi # Return on Investment - used to calculate Buy And Hold strategy
        self.bankroll = bankroll # needed to determine how much we can short
        self.p = p # our principle
        self.q = q # amount of stock
        self.p0 = p # save the initial bankroll for reset
        self.q0 = q
        self.r_buy = r_buy # the ratio of p to spend on purchase
        self.r_sell = r_sell # the ratio of p to liquidate on sell
        self.nt = nt # length of input vector

        self.p_margin = 0 # in development
        self.q_margin = 0


    def buy(self, x, t):
        '''Buy at time index t at value x[t]'''
        p_spent = self.p * self.r_buy
        q_bought = p_spent / x[t]
        self.q += q_bought
        self.p -= p_spent
        return (p_spent, q_bought)


    def sell(self, x, t):
        '''Sell at time index t at value x[t]'''
        q_sold = self.q * self.r_sell
        p_earned = q_sold * x[t]
        self.q -= q_sold
        self.p += p_earned
        return (p_earned, q_sold)


    def short_position(self, x, t1, t2):
        '''The noble short sell. "Borrow" stock in order to sell it immediately, then buy it back at a later date in order return the borrowed shares.
        '''
        q_short = self.bankroll / x[t1] # decide the amount we want to short, since in theory can short infinite
        self.q_margin += q_short
        p_earned = q_short * x[t1]
        self.p += p_earned

        q_return = self.q_margin
        p_buyback = q_return * x[t2]
        self.q_margin -= q_return
        self.p -= p_buyback
        return (p_earned-p_buyback, q_short)


    def liquidate(self, x):
        '''Sell all positions so we are left only with cash'''
        q_sold = self.q
        p_earned = q_sold * x[-1]
        self.q = 0
        self.p += p_earned
        return (p_earned, q_sold)


    def score(self, x):
        '''Liquidate all shares at current market price and then compute how much money we made.
        For the reinforcement paradigm to work well, we define score == 0 as neither gained nor lost any money this round'''
        self.liquidate(x)
        return float((self.roi * self.inflation_loss * self.p) - self.p0)


    def reset(self):
        '''Set funds back to original bankroll'''
        self.p = self.p0
        self.q = self.q0


    def __call__(self, x, *args, **kwargs):
        '''We will use the call protocol to represent running the bot on a single epoch. This will be overloaded for each subclass. Just a fancy little convenience.
            reward = bot()
        would be the same as:
            reward = bot.pullBandit()
        '''
        result = np.random.randn(1)
        if result > 0:
            return 1 #return a positive reward.
        else:
            return -1 #return a negative reward.


class TheWimp(BanditBot):
    name = 'TheWimp'
    # Cowardly refuses to spend any money! Aka baseline
    def __init__(self, nt=50, p=1., q=0., r_buy=1., r_sell=1.):
        super().__init__(nt=nt, p=p, q=q, r_buy=r_buy, r_sell=r_sell)

    def __call__(self, x, *args, **kwargs):
        return self.score(x)


class BuyHold(BanditBot):
    name = 'BuyAndHold'
    # The classic approach - buy and hold a long position
    def __init__(self, nt=50, p=1., q=0., r_buy=1., r_sell=1., roi=1.):
        super().__init__(nt=nt, p=p, q=q, r_buy=r_buy, r_sell=r_sell)
        self.roi = roi

    def __call__(self, x, *args, **kwargs):
        return self.score(x)


class TheBull(BanditBot):
    name = 'TheBull'

    def __call__(self, x, *args, **kwargs):
        self.buy(x, 0)
        return self.score(x)# liquidate full position


class TheBear(BanditBot):
    name = 'TheBear'

    def __call__(self, x, *args, **kwargs):
        self.short_position(x, 0, -1)
        return self.score(x)# liquidate full position



class StratBull(BanditBot):
    name = 'StratBull'
    # Check the overall progress up to the first (fraction) of the epoch. If bullish, buy in
    def __init__(self, nt=50, p=1., q=0., r_buy=1., r_sell=1.):
        super().__init__(nt=nt, p=p, q=q, r_buy=r_buy, r_sell=r_sell)

    def __call__(self, x, *args, **kwargs):
        nt = len(x)
        t1 = nt // 3 # assess period
        if x[t1] > x[0]:
            self.buy(x, t1+1)

        return self.score(x) # liquidate full position


class StratBear(BanditBot):
    name = 'StratBear'
    # Check the overall progress up to the first (fraction) of the epoch. If bearish, short sell
    def __init__(self, nt=50, p=1., q=0., r_buy=1., r_sell=1.):
        super().__init__(nt=nt, p=p, q=q, r_buy=r_buy, r_sell=r_sell)

    def __call__(self, x, *args, **kwargs):
        nt = len(x)
        t1 = nt // 3 # assess period
        if x[t1] < x[0]:
            self.short_position(x, t1+1, -1)

        return self.score(x) # liquidate full position


class StratTwin(BanditBot):
    name = 'StratTwin'
    # Check the overall progress up to the first (fraction) of the epoch. If bull, go long, If bearish, short sell
    def __init__(self, nt=50, p=1., q=0., r_buy=1., r_sell=1.):
        super().__init__(nt=nt, p=p, q=q, r_buy=r_buy, r_sell=r_sell)

    def __call__(self, x, *args, **kwargs):
        nt = len(x)
        t1 = nt // 3 # assess period
        if x[t1] < x[0]:
            self.short_position(x, t1+1, -1)
        else:
            self.buy(x, t1+1)

        return self.score(x) # liquidate full position


class TheMonkey(BanditBot):
    name = 'TheMonkey'
    # A blindfolded monkey throwing darts at the newspaper's financial pages
    def __init__(self, nt=50, p=1., q=0., r_buy=1., r_sell=1.):
        super().__init__(nt=nt, p=p, q=q, r_buy=r_buy, r_sell=r_sell)

    def __call__(self, x, *args, **kwargs):
        nt = len(x)
        a = np.random.randint(0, nt)
        b = np.random.randint(0, nt)
        if a > b:
            a,b = b,a

        self.buy(x, a)
        self.sell(x, b)
        return self.score(x)


class WallStBets(BanditBot):
    name = 'wallstbets YOLO'
    # A slighly more sophisticated monkey
    def __init__(self, nt=50, p=1., q=0., r_buy=1., r_sell=1.):
        super().__init__(nt=nt, p=p, q=q, r_buy=r_buy, r_sell=r_sell)

    def __call__(self, x, *args, **kwargs):
        nt = len(x)
        a = np.random.randint(0, nt)
        b = np.random.randint(0, nt)
        if a > b:
            a,b = b,a

        if np.random.randint(0,2):
            self.short_position(x, a, b)
        else:
            self.buy(x, a)
            self.sell(x, b)
        return self.score(x)
