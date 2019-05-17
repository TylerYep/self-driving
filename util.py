import const

def control_to_string(steer, throttle, high_level):
    return steer * 25, throttle * 30.19, const.CONTROLS[high_level]