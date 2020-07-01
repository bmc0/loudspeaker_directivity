#!/usr/bin/env python

import numpy as np
import sys, getopt

default_interval = 10.
comment_chars = ('#', '*', ';')

def usage():
	print('Usage: {0:s} [OPTION...]'.format(sys.argv[0]))
	print('')
	print('Options:')
	print('    --interval=n          measurement interval in degrees (default: {0:g})'.format(default_interval))
	print('    --mirror-horizontal   use only positive angles for horizontal orbit')
	print('    --mirror-vertical     use only positive angles for vertical orbit')
	print('    --2034                compute ANSI/CTA-2034-A directivity response')
	print('    --use-listening-axis  use listening axis for DI calculations')
	print('    -h, --help            show this help text')

interval = default_interval
mirror_horiz = False
mirror_vert = False
cta_2034 = False
use_listening_axis = False

try:
	optlist, args = getopt.gnu_getopt(sys.argv[1:],
		'h',
		['interval=',
		 'mirror-horizontal',
		 'mirror-vertical',
		 '2034',
		 'use-listening-axis',
		 'help']
	)
except getopt.GetoptError as e:
	print('error: ' + str(e))
	usage()
	sys.exit(2)
for o, a in optlist:
	if o in ('-h', '--help'):
		usage()
		sys.exit()
	elif o == '--interval':
		if 180. % float(a) != 0.:
			print('error: 180 is not a multiple of {0:g}'.format(float(a)))
			sys.exit(2)
		interval = float(a)
	elif o == '--mirror-horizontal':
		mirror_horiz = True
	elif o == '--mirror-vertical':
		mirror_vert = True
	elif o == '--2034':
		cta_2034 = True
	elif o == '--use-listening-axis':
		use_listening_axis = True
	else:
		assert False, 'unhandled option'

if cta_2034 and 10. % interval != 0:
	print('error: interval must go into 10 evenly for ANSI/CTA 2034 calculation.')
	sys.exit(2)

print('Parameters:')
print('  Measurement interval:   {0:g}°'.format(interval))
print('  Number of measurements: {0:d} ({1:d} per orbit)'.format(int(round(360./interval))*2-2, int(round(360./interval))))
print('  Mirror horizontal:      {0:s}'.format(str(mirror_horiz)))
print('  Mirror vertical:        {0:s}'.format(str(mirror_vert)))
print('')

## Returns an array of the form [[angle, weight], ...]
def create_weights(delta):
	angles = np.linspace(0., np.pi, int(round(np.pi / delta)) + 1)
	weights = np.empty_like(angles)

	# Equations from https://www.princeton.edu/3D3A/Publications/Tylka_3D3A_DICalculation.pdf
	def omega(theta):
		return 2. * np.pi * (1. - np.cos(theta))
	def w0():
		# Note: equation in paper expected 0° and 180° to be included in both orbits
		return 1. / (4. * np.pi) * omega(delta / 2.)
	def w1(a):
		return 1. / (4. * np.pi) * ((omega(a + delta / 2.) - omega(a - delta / 2.)) / 4.)

	weights[0] = w0()
	weights[-1] = w0()
	for i in range(1, len(angles)-1):
		weights[i] = w1(angles[i])

	return np.column_stack((angles, weights))

weights = create_weights(np.deg2rad(interval))
#print(np.column_stack((np.rad2deg(weights[:,0]), weights[:,1])))

try:
	freqs = np.loadtxt('h0.txt', comments=comment_chars, usecols=0)
except OSError:
	print('error: failed to read {0:s}'.format("h0.txt"))
	sys.exit(1)
print('Files have {0:d} data points'.format(len(freqs)))

correction = np.zeros_like(freqs)

def read_spl_data(filename):
	print('Reading ' + filename + '...')
	ftmp, data = np.loadtxt(filename, comments=comment_chars, usecols=(0, 1), unpack=True)
	if not np.array_equal(ftmp, freqs):
		print('error: frequencies in {0:s} are not the same as in h0.txt'.format(filename))
		sys.exit(1)
	return data + correction

def try_read_spl_data(filename):
	try:
		return read_spl_data(filename)
	except OSError:
		print('error: failed to read {0:s}'.format(filename))
		sys.exit(1)

try:
	correction = read_spl_data('correction.txt')
except OSError:
	print('correction.txt not present')

def read_partial_orbit(angles, prefix):
	orbit = np.empty((len(angles), len(freqs)))
	for i in range(0, len(angles)):
		orbit[i] = try_read_spl_data('{0:s}{1:g}.txt'.format(prefix, np.rad2deg(angles[i])))
	return orbit

## Note: 0° and 180° are only included in orbit_horiz_positive
orbit_horiz_positive = read_partial_orbit(weights[:,0], 'h')
orbit_horiz_negative = orbit_horiz_positive[1:-1] if mirror_horiz else read_partial_orbit(weights[1:-1,0] * -1, 'h')
orbit_vert_positive = read_partial_orbit(weights[1:-1,0], 'v')
orbit_vert_negative = orbit_vert_positive if mirror_vert else read_partial_orbit(weights[1:-1,0] * -1, 'v')

power_spectrum = np.zeros_like(freqs)

def add_to_power_spectrum(data, weights):
	global power_spectrum
	assert len(data) == len(weights), 'len(data) != len(weights)'
	for i in range(0, len(data)):
		power_spectrum += np.power(10., data[i] / 10.) * weights[i]

add_to_power_spectrum(orbit_horiz_positive, weights[:,1])
add_to_power_spectrum(orbit_horiz_negative, weights[1:-1,1])
add_to_power_spectrum(orbit_vert_positive, weights[1:-1,1])
add_to_power_spectrum(orbit_vert_negative, weights[1:-1,1])

try:
	listening_axis = read_spl_data('listening_axis.txt')
except OSError:
	print('Failed to open listening_axis.txt; using the axial response')
	listening_axis = orbit_horiz_positive[0]

def select_list(x, l):
	r = np.zeros_like(x, dtype=bool)
	for v in l:
		r = r | (x == v)
	return r

def power_average(x):
	return 10. * np.log10(np.mean(np.power(10., x / 10.), axis=0))

if cta_2034:
	listening_window = power_average(np.concatenate((
		orbit_horiz_positive[0:1],
		orbit_horiz_positive[select_list(weights[:,0],    np.deg2rad([10, 20, 30]))],
		orbit_horiz_negative[select_list(weights[1:-1,0], np.deg2rad([10, 20, 30]))],
		orbit_vert_positive[select_list(weights[1:-1,0],  np.deg2rad([10]))],
		orbit_vert_negative[select_list(weights[1:-1,0],  np.deg2rad([10]))],
	)))
	early_reflections = power_average(np.concatenate((
		orbit_vert_negative[select_list(weights[1:-1,0],  np.deg2rad([20, 30, 40]))],          # Floor
		orbit_vert_positive[select_list(weights[1:-1,0],  np.deg2rad([40, 50, 60]))],          # Ceiling
		orbit_horiz_positive[select_list(weights[:,0],    np.deg2rad([0, 10, 20, 30]))],       # Front+
		orbit_horiz_negative[select_list(weights[1:-1,0], np.deg2rad([10, 10, 30]))],          # Front-
		orbit_horiz_positive[select_list(weights[:,0],    np.deg2rad([40, 50, 60, 70, 80]))],  # Side+
		orbit_horiz_negative[select_list(weights[1:-1,0], np.deg2rad([40, 50, 60, 70, 80]))],  # Side-
		orbit_horiz_positive[select_list(weights[:,0],    np.deg2rad([180, 90]))],             # Rear+
		orbit_horiz_negative[select_list(weights[1:-1,0], np.deg2rad([90]))],                  # Rear-
	)))

##
## Plotting
##

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib as mpl
from cycler import cycler

colors = ['#ac3536', '#165c79', '#df8b16', '#333', '#76903e', '#aa5799', '#429b8c', '#888']
mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)

if cta_2034 and not use_listening_axis:
	norm_level = np.median(listening_window[np.logical_and(freqs >= 300, freqs <= 8000)])
else:
	norm_level = np.median(listening_axis[np.logical_and(freqs >= 300, freqs <= 8000)])
print("norm level: {0:g}".format(norm_level))
y_top = int(round(norm_level/5.))*5+20
y_bot = y_top-50
legend_cols = min(8, int(np.ceil(len(weights) / 5.)))

def setup_line_plot():
	fig, ax = plt.subplots()
	fig.set_size_inches(1024./96., 768./96.)
	fig.set_tight_layout(True)

	ax.set_xscale('log')
	ax.set_xlabel('Frequency (Hz)')
	ax.set_ylabel('Sound Pressure Level (dB ref 20µPa)')

	ax.set_xlim(20., 20000.)
	ax.set_ylim(y_bot, y_top)
	ax.grid(linestyle=':', which='both', color='k', alpha=0.2)
	ax.get_yaxis().set_major_locator(tkr.MultipleLocator(base=10))
	ax.get_yaxis().set_minor_locator(tkr.MultipleLocator(2))
	ax.tick_params(which='both', direction='in')

	return fig, ax

## Listening axis, sound power and DI curves

fig, ax = setup_line_plot()

if cta_2034:
	ax.plot(freqs, listening_window, color=colors[4], linestyle='--', label='Listening Window')
ax.plot(freqs, listening_axis, color='k', label='Listening Axis')
if cta_2034:
	ax.plot(freqs, early_reflections, color=colors[1], label='Early Reflections')
ax.plot(freqs, 10. * np.log10(power_spectrum), color=colors[0], label='Total Sound Power')
if cta_2034:
	ax.plot(freqs, (listening_axis if use_listening_axis else listening_window) - early_reflections + y_bot,
		color=colors[1], linestyle='--', label='Early Reflections DI (offset {0:d}dB)'.format(y_bot))
ax.plot(freqs, (listening_window if cta_2034 and not use_listening_axis else listening_axis) - (10. * np.log10(power_spectrum)) + y_bot,
	color=colors[0], linestyle='--', label='{0:s} (offset {1:d}dB)'.format('Sound Power DI' if cta_2034 else 'Directivity Index', y_bot))
#plt.hlines(norm_level, 20., 20000., colors='grey', linestyles=':', label='Calculated {0:s} Level'.format('Listening Window' if cta_2034 else 'Listening Axis'))

if cta_2034:
	ax.set_title('ANSI/CTA-2034-A Directivity Response (DI Ref: {0:s})'.format('Listening Axis' if use_listening_axis else 'Listening Window'))
ax.legend(frameon=False)

print('Writing directivity.png...')
plt.savefig('directivity.png', dpi=96)

## Horizontal curves (positive)

fig, ax = setup_line_plot()

for i in range(0, len(weights)):
	ax.plot(freqs, orbit_horiz_positive[i], label='{0:g}°'.format(np.rad2deg(weights[i][0])))

ax.set_title('Horizontal 0° to 180°')
ax.legend(frameon=False, ncol=legend_cols)

print('Writing directivity_h_pos.png...')
plt.savefig('directivity_h_pos.png', dpi=96)

## Horizontal curves (negative)

fig, ax = setup_line_plot()

for i in range(0, len(weights)):
	ax.plot(freqs, orbit_horiz_positive[i] if i == 0 or i == len(weights)-1 else orbit_horiz_negative[i-1], label='{0:g}°'.format(np.rad2deg(-weights[i][0])))

ax.set_title('Horizontal 0° to -180°')
ax.legend(frameon=False, ncol=legend_cols)

print('Writing directivity_h_neg.png...')
plt.savefig('directivity_h_neg.png', dpi=96)

## Vertical curves (positive)

fig, ax = setup_line_plot()

for i in range(0, len(weights)):
	ax.plot(freqs, orbit_horiz_positive[i] if i == 0 or i == len(weights)-1 else orbit_vert_positive[i-1], label='{0:g}°'.format(np.rad2deg(weights[i][0])))

ax.set_title('Vertical 0° to 180°')
ax.legend(frameon=False, ncol=legend_cols)

print('Writing directivity_v_pos.png...')
plt.savefig('directivity_v_pos.png', dpi=96)

## Vertical curves (negative)

fig, ax = setup_line_plot()

for i in range(0, len(weights)):
	ax.plot(freqs, orbit_horiz_positive[i] if i == 0 or i == len(weights)-1 else orbit_vert_negative[i-1], label='{0:g}°'.format(np.rad2deg(-weights[i][0])))

ax.set_title('Vertical 0° to -180°')
ax.legend(frameon=False, ncol=legend_cols)

print('Writing directivity_v_neg.png...')
plt.savefig('directivity_v_neg.png', dpi=96)

## Contour plots

def do_contour_plot(x, y, z, title, filename):
	z_range=[-24, 6]

	fig, ax = plt.subplots()
	fig.set_size_inches(1024./96., 768./96.)
	fig.set_tight_layout(True)

	cs = ax.contourf(
		x, y, z,
		levels=np.linspace(z_range[0], z_range[1], z_range[1]-z_range[0]+1),
		extend='both',
		cmap=plt.cm.magma,
	)
	cs.cmap.set_over('w')
	cs2 = ax.contour(
		cs,
		levels=cs.levels[::2],
		colors='k',
		linewidths=0.7
	)

	cbar = fig.colorbar(cs, pad=0.02, fraction=0.06)
	cbar.add_lines(cs2)

	ax.set_title(title)
	ax.set_xscale('log')
	ax.set_xlabel('Frequency (Hz)')
	ax.set_ylabel('Angle (degrees)')
	cbar.ax.set_ylabel('Sound Pressure Level (dB norm)')

	ax.set_xlim(100., 20000.)
	ax.set_ylim(-180., 180.)
	ax.grid(linestyle=':', which='both', color='k', alpha=0.2)
	ax.get_yaxis().set_major_locator(tkr.MultipleLocator(base=30))
	ax.get_yaxis().set_minor_locator(tkr.MultipleLocator(10))

	print('Writing ' + filename + '...')
	plt.savefig(filename, dpi=96)

do_contour_plot(
	freqs,
	np.rad2deg(np.concatenate((
		np.flip(weights[1:,0] * -1., 0),
		weights[:,0]
	))),
	np.concatenate((
		orbit_horiz_positive[-1:],
		np.flip(orbit_horiz_negative, 0),
		orbit_horiz_positive
	)) - norm_level,
	'Horizontal Directivity',
	'directivity_contour_h.png',
)

do_contour_plot(
	freqs,
	np.rad2deg(np.concatenate((
		np.flip(weights[1:,0] * -1., 0),
		weights[:,0]
	))),
	np.concatenate((
		orbit_horiz_positive[-1:],
		np.flip(orbit_vert_negative, 0),
		orbit_horiz_positive[0:1],
		orbit_vert_positive,
		orbit_horiz_positive[-1:],
	)) - norm_level,
	'Vertical Directivity',
	'directivity_contour_v.png',
)
