from pylab import *

class RangeError(Exception):
    def __init__(self,s=None):
        Exception.__init__(self,s)

def prepare_line(line,pad=16):
    """Prepare a line for recognition; this inverts it, transposes
    it, and pads it."""
    line = line * 1.0/amax(line)
    line = amax(line)-line
    line = line.T
    if pad>0:
        w = line.shape[1]
        line = vstack([zeros((pad,w)),line,zeros((pad,w))])
    return line

def randu(*shape):
    # ATTENTION: whether you use randu or randn can make a difference.
    """Generate uniformly random values in the range (-1,1).
    This can usually be used as a drop-in replacement for `randn`
    resulting in a different distribution."""
    return 2*rand(*shape)-1

def check_nan(*args,**kw):
    "Check whether there are any NaNs in the argument arrays."
    for arg in args:
        if isnan(arg).any():
            raise FloatingPointError()


class Network:
    """General interface for networks. This mainly adds convenience
    functions for `predict` and `train`."""
    def predict(self,xs):
        return self.forward(xs)

class Softmax(Network):
    """A logistic regression network."""
    def forward(self,ys):
        n = len(ys)
        inputs,zs = [None]*n,[None]*n
        for i in range(n):
            inputs[i] = concatenate([ones(1),ys[i]])
            temp = dot(self.W2,inputs[i])
            temp = exp(clip(temp,-100,100))
            temp /= sum(temp)
            zs[i] = temp
        self.state = (inputs,zs)
        return zs

# These are the nonlinearities used by the LSTM network.
# We don't bother parameterizing them here

def ffunc(x):
    "Nonlinearity used for gates."
    return 1.0/(1.0+exp(-x))
def fprime(x,y=None):
    "Derivative of nonlinearity used for gates."
    if y is None: y = sigmoid(x)
    return y*(1.0-y)
def gfunc(x):
    "Nonlinearity used for input to state."
    return tanh(x)
def gprime(x,y=None):
    "Derivative of nonlinearity used for input to state."
    if y is None: y = tanh(x)
    return 1-y**2
# ATTENTION: try linear for hfunc
def hfunc(x):
    "Nonlinearity used for output."
    return tanh(x)
def hprime(x,y=None):
    "Derivative of nonlinearity used for output."
    if y is None: y = tanh(x)
    return 1-y**2

class LSTM(Network):
    """A standard LSTM network. This is a direct implementation of all the forward
    and backward propagation formulas, mainly for speed. (There is another, more
    abstract implementation as well, but that's significantly slower in Python
    due to function call overhead.)"""
    def reset(self,n):
        """Reset the contents of the internal state variables to `nan`"""
        vars = "cix ci gix gi gox go gfx gf"
        vars += " state output gierr gferr goerr cierr stateerr outerr"
        vars += " source sourceerr"
        for v in vars.split():
            getattr(self,v)[:,:] = nan
    def forward(self,xs):
        """Perform forward propagation of activations."""
        ni,ns,na = self.dims
        assert len(xs[0])==ni
        n = len(xs)
        if n>len(self.gi):
            raise "input too large for LSTM model"
        self.last_n = n
        self.reset(n)
        for t in range(n):
            prev = zeros(ns) if t==0 else self.output[t-1]
            self.source[t,0] = 1
            self.source[t,1:1+ni] = xs[t]
            self.source[t,1+ni:] = prev
            dot(self.WGI,self.source[t],out=self.gix[t])
            dot(self.WGF,self.source[t],out=self.gfx[t])
            dot(self.WGO,self.source[t],out=self.gox[t])
            dot(self.WCI,self.source[t],out=self.cix[t])
            if t>0:
                # ATTENTION: peep weights are diagonal matrices
                self.gix[t] += self.WIP*self.state[t-1]
                self.gfx[t] += self.WFP*self.state[t-1]
            self.gi[t] = ffunc(self.gix[t])
            self.gf[t] = ffunc(self.gfx[t])
            self.ci[t] = gfunc(self.cix[t])
            self.state[t] = self.ci[t]*self.gi[t]
            if t>0:
                self.state[t] += self.gf[t]*self.state[t-1]
                self.gox[t] += self.WOP*self.state[t]
            self.go[t] = ffunc(self.gox[t])
            self.output[t] = hfunc(self.state[t]) * self.go[t]
        assert not isnan(self.output[:n]).any()
        return self.output[:n]

################################################################
# combination classifiers
################################################################


class Stacked(Network):
    """Stack two networks on top of each other."""
    def forward(self,xs):
        for i,net in enumerate(self.nets):
            xs = net.forward(xs)
        return xs

class Reversed(Network):
    """Run a network on the time-reversed input."""
    def forward(self,xs):
        return self.net.forward(xs[::-1])[::-1]

class Parallel(Network):
    """Run multiple networks in parallel on the same input."""
    def forward(self,xs):
        outputs = [net.forward(xs) for net in self.nets]
        outputs = zip(*outputs)
        outputs = [concatenate(l) for l in outputs]
        return outputs

################################################################
# LSTM classification with forward/backward alignment ("CTC")
################################################################


from scipy.ndimage import measurements,filters

def translate_back(outputs,threshold=0.7):
    """Translate back. Thresholds on class 0, then assigns
    the maximum class to each region."""
    labels,n = measurements.label(outputs[:,0]<threshold)
    mask = tile(labels.reshape(-1,1),(1,outputs.shape[1]))
    maxima = measurements.maximum_position(outputs,mask,arange(1,amax(mask)+1))
    return [c for (r,c) in maxima]


def normalize_nfkc(s):
    print "fuck"
    return unicodedata.normalize('NFKC',s)

class SeqRecognizer:
    """Perform sequence recognition using BIDILSTM and alignment."""
    def predictSequence(self,xs):
        "Predict an integer sequence of codes."
        assert xs.shape[1]==self.Ni,"wrong image height (image: %d, expected: %d)"%(xs.shape[1],self.Ni)
        self.outputs = array(self.lstm.forward(xs))
        return translate_back(self.outputs)
    def l2s(self,l):
        "Convert a code sequence into a unicode string after recognition."
        l = self.codec.decode(l)
        return u"".join(l)
    def predictString(self,xs):
        "Predict output as a string. This uses codec and normalizer."
        cs = self.predictSequence(xs)
        return self.l2s(cs)

class Codec:
    """Translate between integer codes and characters."""
    def decode(self,l):
        "Decode a code sequence into a string."
        s = [self.code2char.get(c,"~") for c in l]
        return s