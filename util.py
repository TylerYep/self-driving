import const

def control_to_string(steer, throttle, high_level):
    return steer, throttle, const.CONTROLS[high_level]