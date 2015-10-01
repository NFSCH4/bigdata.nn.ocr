from pylab import *
import cPickle, gzip, re


def write_text(file,s):
    """Write the given string s to the output file."""
    with open(file,"w") as stream:
        if type(s)==unicode: s = s.encode("utf-8")
        stream.write(s)

def allsplitext(path):
    """Split all the pathname extensions, so that "a/b.c.d" -> "a/b", ".c.d" """
    match = re.search(r'((.*/)*[^.]*)([^/]*)',path)
    if not match:
        return path,""
    else:
        return match.group(1),match.group(3)

def unpickle_find_global(mname,cname):
    if mname=="lstm.lstm":
        import lstm
        return getattr(lstm,cname)
    if not mname in sys.modules.keys():
        exec "import "+mname
    return getattr(sys.modules[mname],cname)

def load_object(fname,zip=0,nofind=0,verbose=0):
    """Loads an object from disk."""
    with gzip.GzipFile(fname,"rb") as stream:
        unpickler = cPickle.Unpickler(stream)
        unpickler.find_global = unpickle_find_global
        return unpickler.load()

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
