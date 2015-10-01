## DUMMY ##
class CenterNormalizer:
    def __init__(self,target_height=48,params=(4,1.0,0.3)):
        self.debug = int(os.getenv("debug_center") or "0")
        self.target_height = target_height
        self.range,self.smoothness,self.extra = params
        print "# CenterNormalizer"

