
v0 = 5
g = 9.81
t = 0.6
y = v0 * t - ((1.0/2) * g * t**2)
print y

print "at initial speed of %g m/s, we get vertical reading of %.2f at %g seconds" % (v0, y, t)

print "at initial speed of {v:g} m/s, we get vertical reading of {y:.2f} at {t:g} seconds".format(t = t, v = v0, y = y)

