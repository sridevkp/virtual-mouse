import time
from functools import partial, wraps

class TooSoon(Exception):
  """Can't be called so soon"""
  pass

class CoolDownDecorator(object):
  def __init__(self,func,interval):
    self.func = func
    self.interval = interval
    self.last_run = 0
  def __get__(self,obj,objtype=None):
    if obj is None:
      return self.func
    return partial(self,obj)
  def __call__(self,*args,**kwargs):
    now = time.time()
    if now - self.last_run < self.interval:
      pass
    #   print("Call after {0} seconds".format(self.last_run + self.interval - now))
    #   raise TooSoon("Call after {0} seconds".format(self.last_run + self.interval - now))
    else:
      self.last_run = now
      return self.func(*args,**kwargs)

def CoolDown(interval):
  def applyDecorator(func):
    decorator = CoolDownDecorator(func=func,interval=interval)
    return wraps(func)(decorator)
  return applyDecorator