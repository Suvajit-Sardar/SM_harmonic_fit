# Radial velocity fitting

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# STYLE SETTINGS (Matched to Rotation Curve Script)
custom_rcparams = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "font.size": 30,
    "axes.labelsize": 30,
    "axes.titlesize": 30,
    "legend.fontsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "xtick.major.pad": 10,
    "ytick.major.pad": 2,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "lines.markersize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "xtick.bottom": True,
    "ytick.left": True,
    "ytick.right": True,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "figure.figsize": (10, 8),
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "black",
    "legend.loc": "best",
    "axes.edgecolor": "black",
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    "text.usetex": False,
}
mpl.rcParams.update(custom_rcparams)


## INPUT PARAMS
fits_file_mom1 = '20_tar_mom1.fits'
fits_file_mom2 = '20_tar_mom2.fits'

x_c, y_c = 24.53919445613506, 26.309980891051914  
segments = 24            
inc_deg = 49              
V_sys = 4676.0              
pix_scale = 4.0             

PA_deg = 53 + 90            
KPC_PER_ARCSEC = 0.329

# Height control for A/R text labels 
text_label_height = 0.55

# Rings in arcsec
rings_arcsec = [
    (38.15, 47.85),
    (47.85, 57.55),
    (57.55, 67.25),
    (67.25, 76.95)
]
rings_pix = [(r_in / pix_scale, r_out / pix_scale) for r_in, r_out in rings_arcsec]

# TRM vrot
vrot_bbarolo = [260.175, 295.133, 301.112, 301.181]

# PLOT CONTROLS
inset_pos = [0.614, 0.13, 0.30, 0.30] 
inset_zoom_pix = 25  


### READ FITS & CONVERT UNITS
hdul1 = fits.open(fits_file_mom1)
data_mom1 = np.squeeze(hdul1[0].data)
data_mom1 = data_mom1 / 1000.0  

hdul2 = fits.open(fits_file_mom2)
data_mom2 = np.squeeze(hdul2[0].data)
data_mom2 = data_mom2 / 1000.0  


#### GENERATE DEPROJECTED COORDINATES & AZIMUTH
y_indices, x_indices = np.indices(data_mom1.shape)

dx = x_indices - x_c
dy = y_indices - y_c

PA_rad = np.radians(PA_deg)
inc_rad = np.radians(inc_deg)

dx_rot = dx * np.cos(PA_rad) + dy * np.sin(PA_rad)
dy_rot = -dx * np.sin(PA_rad) + dy * np.cos(PA_rad)

dy_deproj = dy_rot / np.cos(inc_rad)
R_pix = np.sqrt(dx_rot**2 + dy_deproj**2)

theta_rad = np.arctan2(dy_deproj, dx_rot)
theta_deg = np.degrees(theta_rad) % 360.0


##### PREPARE DATA FOR HARMONIC FITTING
v_plane_map = (data_mom1 - V_sys) / np.sin(inc_rad)
v_err_map = data_mom2  

def harmonic_model(theta, v_rot, v_rad):
    return v_rot * np.cos(theta) + v_rad * np.sin(theta)


###### CALCULATE WEIGHTED FITS & BINNED PROFILES
theta_edges = np.linspace(0, 360, segments + 1)
theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0

binned_profiles = []
fit_curves = []

# Arrays for vrad plot
v_rad_fits_array = []
rms_array = []
ring_centers_arcsec = [np.mean(r) for r in rings_arcsec]

# New arrays for A and R fits
v_rad_app_array = []
rms_app_array = []
v_rad_rec_array = []
rms_rec_array = []

# --- Determine Masks for Approaching and Receding Sides ---
sky_phi = np.degrees(np.arctan2(dy, dx)) % 360.0
# Receding:
mask_rec_sky = (sky_phi >= 134) & (sky_phi < 314)
# Approaching:
mask_app_sky = (sky_phi >= 314) | (sky_phi < 134)

total_chi2 = 0.0
total_dof = 0

print("\n" + "-"*100)
print(f"{'Ring (arcsec)':<15} | {'Fixed_Vrot':<10} | {'Fit_Vrad':<10} | {'RMS (km/s)':<10} | {'DoF':<8} | {'Chi2':<10} | {'Red_Chi2':<8}")
print("-" * 100)

for i, (r_in_pix, r_out_pix) in enumerate(rings_pix):
    mask_ring = (R_pix >= r_in_pix) & (R_pix < r_out_pix) & np.isfinite(v_plane_map) & np.isfinite(v_err_map) & (v_err_map >= 5.0)
    
    # ---------------- OVERALL FIT ----------------
    val_theta_rad = theta_rad[mask_ring]
    val_v_plane = v_plane_map[mask_ring]
    val_v_err = v_err_map[mask_ring]
    
    fixed_vrot = vrot_bbarolo[i]
    p0 = [0.0] 
    
    popt, pcov = curve_fit(
        lambda t, v_rad: harmonic_model(t, fixed_vrot, v_rad), 
        val_theta_rad, val_v_plane, p0=p0, sigma=val_v_err, absolute_sigma=True     
    )
    v_rad_fit = popt[0]
    residuals = val_v_plane - harmonic_model(val_theta_rad, fixed_vrot, v_rad_fit)
    rms = np.std(residuals)
    
    v_rad_fits_array.append(v_rad_fit)
    rms_array.append(rms)
    
    # ---------------- APPROACHING FIT ----------------
    mask_app = mask_ring & mask_app_sky
    val_theta_app = theta_rad[mask_app]
    val_v_plane_app = v_plane_map[mask_app]
    val_v_err_app = v_err_map[mask_app]
    
    if len(val_theta_app) > 5:
        popt_app, _ = curve_fit(
            lambda t, v_rad: harmonic_model(t, fixed_vrot, v_rad), 
            val_theta_app, val_v_plane_app, p0=[0.0], sigma=val_v_err_app, absolute_sigma=True
        )
        v_rad_fit_app = popt_app[0]
        rms_app = np.std(val_v_plane_app - harmonic_model(val_theta_app, fixed_vrot, v_rad_fit_app))
    else:
        v_rad_fit_app, rms_app = np.nan, np.nan
        
    v_rad_app_array.append(v_rad_fit_app)
    rms_app_array.append(rms_app)
    
    # ---------------- RECEDING FIT ----------------
    mask_rec = mask_ring & mask_rec_sky
    val_theta_rec = theta_rad[mask_rec]
    val_v_plane_rec = v_plane_map[mask_rec]
    val_v_err_rec = v_err_map[mask_rec]
    
    if len(val_theta_rec) > 5:
        popt_rec, _ = curve_fit(
            lambda t, v_rad: harmonic_model(t, fixed_vrot, v_rad), 
            val_theta_rec, val_v_plane_rec, p0=[0.0], sigma=val_v_err_rec, absolute_sigma=True
        )
        v_rad_fit_rec = popt_rec[0]
        rms_rec = np.std(val_v_plane_rec - harmonic_model(val_theta_rec, fixed_vrot, v_rad_fit_rec))
    else:
        v_rad_fit_rec, rms_rec = np.nan, np.nan
        
    v_rad_rec_array.append(v_rad_fit_rec)
    rms_rec_array.append(rms_rec)
    
    # Statistics
    dof = len(val_v_plane) - 1  
    chi2 = np.sum((residuals / val_v_err)**2)
    red_chi2 = chi2 / dof if dof > 0 else np.nan
    
    total_chi2 += chi2
    total_dof += dof
    
    print(f"{rings_arcsec[i][0]:5.2f}'' - {rings_arcsec[i][1]:5.2f}'' | {fixed_vrot:10.1f} | {v_rad_fit:10.1f} | {rms:10.1f} | {dof:8d} | {chi2:10.1f} | {red_chi2:8.2f}")
    
    theta_smooth = np.radians(np.linspace(0, 360, 200))
    fit_curves.append(harmonic_model(theta_smooth, fixed_vrot, v_rad_fit))
    
    v_ring_binned = np.zeros(segments)
    for j in range(segments):
        t_min = theta_edges[j]
        t_max = theta_edges[j+1]
        mask_wedge = mask_ring & (theta_deg >= t_min) & (theta_deg < t_max)
        if np.sum(mask_wedge) > 0:
            v_ring_binned[j] = np.nanmean(v_plane_map[mask_wedge])
        else:
            v_ring_binned[j] = np.nan
            
    binned_profiles.append(v_ring_binned)

print("-" * 100)
overall_red_chi2 = total_chi2 / total_dof if total_dof > 0 else np.nan
print(f"{'OVERALL MODEL':<15} | {'-':<10} | {'-':<10} | {'-':<10} | {total_dof:8d} | {total_chi2:10.1f} | {overall_red_chi2:8.2f}")
print("-"*100 + "\n")


### AZIMUTHAL PROFILE
fig, ax1 = plt.subplots() 

# Shading
ax1.axvspan(90, 270, facecolor='#F7F8FF', alpha=0.8, zorder=0)  
ax1.axvspan(270, 450, facecolor='#FFF7F7', alpha=0.8, zorder=0) 


ax1.text(180, text_label_height, 'Approaching', color='darkblue', ha='center', va='top', 
         fontweight='bold', fontsize=22, transform=ax1.get_xaxis_transform(), zorder=10)
ax1.text(360, text_label_height, 'Receding', color='darkred', ha='center', va='top', 
         fontweight='bold', fontsize=22, transform=ax1.get_xaxis_transform(), zorder=10)

colors = ['darkblue', 'magenta', 'green', 'red']
markers = ['o', 's', '^', 'D']
theta_smooth_deg = np.linspace(0, 360, 200)

for i, (binned_data, fit_curve, arcsec_bounds) in enumerate(zip(binned_profiles, fit_curves, rings_arcsec)):
    label_str = f"{arcsec_bounds[0]}'' - {arcsec_bounds[1]}''"
    
    # Shift data points
    theta_centers_shifted = np.where(theta_centers < 90, theta_centers + 360, theta_centers)
    sort_idx = np.argsort(theta_centers_shifted)
    tc_plot = theta_centers_shifted[sort_idx]
    bd_plot = binned_data[sort_idx]
    
    # Shift curves
    theta_smooth_shifted = np.where(theta_smooth_deg < 90, theta_smooth_deg + 360, theta_smooth_deg)
    sort_smooth_idx = np.argsort(theta_smooth_shifted)
    ts_plot = theta_smooth_shifted[sort_smooth_idx]
    fc_plot = fit_curve[sort_smooth_idx]
    
    ax1.plot(tc_plot, bd_plot, marker=markers[i], linestyle='None', color=colors[i], markersize=8, 
             mfc='none', alpha=1, zorder=3)
    
    ax1.plot(ts_plot, fc_plot, linestyle='--', color=colors[i], 
             linewidth=2.0, label=label_str, zorder=2, alpha =0.6)

ax1.set_xlabel(r'Azimuth [$^{\circ}$]')
ax1.set_ylabel(r'$(V_{\rm LOS} - V_{\rm sys}) / \sin(i)$ [km s$^{-1}$]')

leg = ax1.legend(loc='upper left', framealpha=0.9, fontsize=22)
leg.set_zorder(5)

ax1.set_xlim(90, 450)
ticks = np.arange(90, 451, 45)
ax1.set_xticks(ticks)
ax1.set_xticklabels([f"{t}" if t <= 360 else f"{t - 360}" for t in ticks])
ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.8)

ax1.xaxis.set_minor_locator(AutoMinorLocator(5)) 
ax1.yaxis.set_minor_locator(AutoMinorLocator(5))

# Inset map
ax_inset = fig.add_axes(inset_pos)
vmax_disp = np.nanmax(np.abs(data_mom1 - V_sys))
ax_inset.imshow(data_mom1, origin='lower', cmap='RdBu_r', 
                vmin=V_sys - vmax_disp, vmax=V_sys + vmax_disp)

all_edges_pix = set([r[0] for r in rings_pix] + [r[1] for r in rings_pix])
for r_edge in all_edges_pix:
    ellipse = patches.Ellipse(
        xy=(x_c, y_c), width=2 * r_edge, height=2 * r_edge * np.cos(inc_rad), 
        angle=PA_deg, edgecolor='black', facecolor='none', 
        linewidth=0.8, alpha=0.6, linestyle=':'
    )
    ax_inset.add_patch(ellipse)

spike_angles = np.arange(0, 360, 45)
spike_length = rings_pix[-1][1]  

for ang in spike_angles:
    ang_rad = np.radians(ang)
    dx_rot_spike = spike_length * np.cos(ang_rad)
    dy_deproj_spike = spike_length * np.sin(ang_rad)
    dy_rot_spike = dy_deproj_spike * np.cos(inc_rad)
    
    dx_spike = dx_rot_spike * np.cos(PA_rad) - dy_rot_spike * np.sin(PA_rad)
    dy_spike = dx_rot_spike * np.sin(PA_rad) + dy_rot_spike * np.cos(PA_rad)
    
    ax_inset.plot([x_c, x_c + dx_spike], [y_c, y_c + dy_spike], color='black', linewidth=1, linestyle='-', alpha=0.8)
    text_pad = 1.25
    ax_inset.text(x_c + dx_spike * text_pad, y_c + dy_spike * text_pad, f"{ang}°", 
                  color='black', fontsize=22, ha='center', va='center', weight='bold')

ax_inset.plot(x_c, y_c, marker='+', color='black', markersize=8)
ax_inset.set_xlim(x_c - inset_zoom_pix, x_c + inset_zoom_pix)
ax_inset.set_ylim(y_c - inset_zoom_pix, y_c + inset_zoom_pix)
ax_inset.tick_params(axis='both', which='both', direction='in', color='black', length=3)
ax_inset.set_xticklabels([])
ax_inset.set_yticklabels([])

for spine in ax_inset.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

plt.savefig("20_target_harmonic_fit_azimuthal.pdf", bbox_inches='tight')






### RADIAL VELOCITY PROFILE

fig2, ax_vrad = plt.subplots(figsize=(10, 8))
rc_arcsec_arr = np.array(ring_centers_arcsec)

# 1. Harmonic fit (A - Blue)
v_rad_outflow_app = np.abs(v_rad_app_array)
ax_vrad.errorbar(rc_arcsec_arr, v_rad_outflow_app, yerr=rms_app_array, 
                 fmt='s', color='blue', markeredgecolor='black', label='Approaching', 
                 capsize=4, elinewidth=1.5, zorder=5)

# 2. Harmonic fit (B - Green)
v_rad_outflow = np.abs(v_rad_fits_array)
ax_vrad.errorbar(rc_arcsec_arr, v_rad_outflow, yerr=rms_array, 
                 fmt='o', color='green', markeredgecolor='black', label='Both', 
                 capsize=4, elinewidth=1.5, zorder=6)

# 3.  Harmonic fit (R - Red)
v_rad_outflow_rec = np.abs(v_rad_rec_array)
ax_vrad.errorbar(rc_arcsec_arr, v_rad_outflow_rec, yerr=rms_rec_array, 
                 fmt='^', color='red', markeredgecolor='black', label='Receding', 
                 capsize=4, elinewidth=1.5, zorder=5)

# 4. TRM Fit Values
trm_vrad = np.array([np.nan, 60.000, 60.000, np.nan])
trm_err_lower = [np.nan, 57.054, 46.511, np.nan]
trm_err_upper = [np.nan, 65.422, 52.001, np.nan]
trm_yerr = [trm_err_lower, trm_err_upper]


# ax_vrad.errorbar(rc_arcsec_arr, trm_vrad, yerr=trm_yerr, 
#                  fmt='s', color='#8A0081', label='TRM fit', 
#                  capsize=5, capthick=2, markersize=10, elinewidth=2, zorder=4)

# VERTICAL LINES & SHADING 
val1_arcsec = 73.2228 / 2
val2_arcsec = 148 / 2

ax_vrad.axvspan(val1_arcsec, val2_arcsec, color='lightgray', alpha=0.5, zorder=0)
ax_vrad.axvline(val1_arcsec, color='gray', linestyle='--', linewidth=1.5, zorder=1)
ax_vrad.axvline(val2_arcsec, color='gray', linestyle='--', linewidth=1.5, zorder=1)

ax_vrad.set_xlim(0, 80.5)

ax_vrad.set_ylim(bottom=0)
_, ymax = ax_vrad.get_ylim()
ax_vrad.set_ylim(-18, ymax + 10) 

ax_vrad.set_ylabel(r'$V_{\rm rad}$ [km s$^{-1}$]')
ax_vrad.set_xlabel('Radius [arcsec]')
ax_vrad.tick_params(axis='y', labelrotation=90)

# Right Y-Axis
ax_right = ax_vrad.twinx()
ax_right.set_ylim(ax_vrad.get_ylim())
ax_right.set_yticklabels([]) 
ax_right.minorticks_on()
ax_right.tick_params(direction='in')

# Top X-Axis
ax_top = ax_vrad.twiny()
xlim_as = ax_vrad.get_xlim()
xlim_kpc = (xlim_as[0] * KPC_PER_ARCSEC, xlim_as[1] * KPC_PER_ARCSEC)
ax_top.set_xlim(xlim_kpc)
ax_top.set_xlabel('Radius [kpc]', labelpad=10)
ax_top.minorticks_on()
ax_vrad.minorticks_on()


handles, labels = ax_vrad.get_legend_handles_labels()
ax_vrad.legend(handles, labels, loc='lower left', ncol=1, fontsize=25)

plt.tight_layout()
plt.savefig("20_target_vrad_profile.pdf", bbox_inches='tight')

plt.show()
