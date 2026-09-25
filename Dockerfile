FROM alpine:latest

# Environment setup
ENV MOZ_ENABLE_WAYLAND=1
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
ENV WAYLAND_DISPLAY=wayland-0
ENV HOMEPAGE_URL="https://signage-demo.admrl.co/admiral.html"

# Options for AUTO_FULLSCREEN: "true" (fullscreen/maximized), "false" (normal window)
ENV AUTO_FULLSCREEN="true"

# Install Weston, seatd, eudev (for udev), Mesa 3D, VA-API Drivers, FFmpeg, pciutils, and Firefox
RUN ARCH=$(uname -m) && \
    EXTRA_PKGS="" && \
    if [ "$ARCH" = "x86_64" ]; then \
        EXTRA_PKGS="intel-media-driver libva-intel-driver"; \
    fi && \
    apk add --no-cache \
        weston \
        weston-backend-drm \
        weston-shell-desktop \
        weston-clients \
        weston-terminal \
        seatd \
        eudev \
        mesa-gbm \
        mesa-dri-gallium \
        mesa-va-gallium \
        mesa-egl \
        libva \
        ffmpeg-libs \
        libdrm \
        pciutils \
        dbus \
        dbus-x11 \
        adwaita-icon-theme \
        firefox-esr \
        ttf-dejavu \
        bash \
        $EXTRA_PKGS

# Configure Weston: idle-time=0 (No Sleep), 1080p mode, panel, background
RUN mkdir -p /etc/xdg/weston && printf '[core]\n\
backend=drm-backend.so\n\
socket-name=wayland-0\n\
idle-time=0\n\
\n\
[shell]\n\
background-color=0xff1e222a\n\
panel-position=bottom\n\
clock-format=24h\n\
allow-zap=true\n\
\n\
[output]\n\
name=HDMI-A-1\n\
mode=1920x1080@60\n\
\n\
[output]\n\
name=HDMI-A-2\n\
mode=1920x1080@60\n' > /etc/xdg/weston/weston.ini

# Add our sandbox rules to ignore non-UI hardware (prevents reloading host disk/network controllers)
RUN mkdir -p /etc/udev/rules.d && \
    echo 'ACTION=="add|change", SUBSYSTEM!="drm|graphics|input|tty", OPTIONS:="ignore"' > /etc/udev/rules.d/00-container-sandbox.rules

# Startup script generator
RUN cat << 'EOF' > /start.sh
#!/bin/bash
export XDG_RUNTIME_DIR=/tmp/runtime-root
export MOZ_ENABLE_WAYLAND=1
export MOZ_WEBRENDER=1
export WAYLAND_DISPLAY=wayland-0
export NO_AT_BRIDGE=1

# 1. Clean and prepare XDG_RUNTIME_DIR
rm -rf "$XDG_RUNTIME_DIR"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 0700 "$XDG_RUNTIME_DIR"

# 2. Configure Firefox Policy (Homepage + Memory Optimization + Hardware Accel)
mkdir -p /usr/lib/firefox-esr/distribution
WEBRENDER_ALL="true"
WEBRENDER_SOFTWARE="false"
if [ "$GFX_WEBRENDER_SOFTWARE" = "true" ]; then
    WEBRENDER_ALL="false"
    WEBRENDER_SOFTWARE="true"
fi
cat << POLICIES > /usr/lib/firefox-esr/distribution/policies.json
{
  "policies": {
    "Homepage": {
      "URL": "${HOMEPAGE_URL:-https://duckduckgo.com}",
      "Locked": true,
      "StartPage": "homepage"
    },
    "OverrideFirstRunPage": "",
    "OverridePostUpdatePage": "",
    "UserMessaging": {
      "SkipOnboarding": true
    },
    "Preferences": {
      "media.ffmpeg.vaapi.enabled": true,
      "gfx.webrender.all": ${WEBRENDER_ALL},
      "gfx.webrender.software": ${WEBRENDER_SOFTWARE},
      "dom.ipc.processCount": 1,
      "dom.ipc.processCount.webIsolated": 1,
      "browser.tabs.remote.separatePrivilegedContentProcess": false,
      "media.rdd-process.enabled": false,
      "media.utility-process.enabled": false,
      "browser.cache.memory.capacity": 32768,
      "browser.cache.disk.enable": false,
      "image.mem.surfacecache.max_size_kb": 32768,
      "browser.sessionhistory.max_entries": 2,
      "browser.sessionstore.max_tabs_undo": 0,
      "javascript.options.mem.max": 65536
    }
  }
}
POLICIES

# 2b. Configure Architecture & Memory Preferences (Requires defaults/pref/ JS file)
mkdir -p /usr/lib/firefox-esr/defaults/pref
cat << 'PREFS' > /usr/lib/firefox-esr/defaults/pref/admiral.js
pref("gfx.webrender.all", true);
pref("gfx.webrender.software", false);
pref("media.ffmpeg.vaapi.enabled", true);
pref("fission.autostart", false);
pref("fission.webContentIsolationStrategy", 0);
pref("dom.ipc.processCount", 1);
pref("dom.ipc.processCount.webIsolated", 1);
pref("network.process.enabled", false);
pref("media.rdd-process.enabled", false);
pref("media.utility-process.enabled", false);
pref("browser.tabs.remote.separatePrivilegedContentProcess", false);
pref("browser.tabs.remote.separatePrivilegedMozillaWebContentProcess", false);
pref("extensions.webextensions.remote", false);
pref("browser.cache.memory.capacity", 32768);
pref("browser.cache.disk.enable", false);
pref("image.mem.surfacecache.max_size_kb", 32768);
pref("browser.sessionhistory.max_entries", 2);
pref("browser.sessionstore.max_tabs_undo", 0);
pref("javascript.options.mem.max", 65536);
PREFS

# 3. Pre-seed Firefox Profile Window Geometry (Prevents 0x0 Fullscreen Crash)
mkdir -p /root/.mozilla/firefox/profile.default
cat << PROFILES > /root/.mozilla/firefox/profiles.ini
[Profile0]
Name=default
IsRelative=1
Path=profile.default
Default=1
[General]
StartWithLastProfile=1
Version=2
PROFILES
SIZEMODE="maximized"
if [ "$AUTO_FULLSCREEN" = "true" ]; then
    SIZEMODE="fullscreen"
fi
cat << XULSTORE > /root/.mozilla/firefox/profile.default/xulstore.json
{
  "chrome://browser/content/browser.xhtml": {
    "main-window": {
      "sizemode": "${SIZEMODE}",
      "width": "1920",
      "height": "1080",
      "screenX": "0",
      "screenY": "0"
    }
  }
}
XULSTORE

# 4. Setup UDEV Environment (Critical for Seatd & DRM discovery)
echo "Starting udevd..."
/sbin/udevd --daemon
echo "Triggering udev for UI devices..."
udevadm trigger --action=add --subsystem-match=drm --subsystem-match=graphics --subsystem-match=input --subsystem-match=tty
udevadm settle

# 5. Setup DBus
echo "Starting DBus..."
mkdir -p /run/dbus
dbus-uuidgen --ensure
dbus-daemon --system --fork
eval $(dbus-launch --sh-syntax)

# 6. Start seatd daemon
echo "Starting seatd..."
seatd -u root &
sleep 1

# 7. Start Weston Compositor
echo "Starting Weston..."
weston --debug --socket=wayland-0 --config=/etc/xdg/weston/weston.ini &

# 8. Wait for Wayland display socket
echo "Waiting for Wayland display socket ($WAYLAND_DISPLAY)..."
while [ ! -S "$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY" ]; do
    sleep 0.5
done
sleep 1
echo "Wayland desktop ready!"

# 9. Launch Firefox with explicit profile and kiosk fullscreen if requested
EXTRA_ARGS=""
if [ "$AUTO_FULLSCREEN" = "true" ]; then
    EXTRA_ARGS="--kiosk"
fi
exec firefox-esr --profile /root/.mozilla/firefox/profile.default $EXTRA_ARGS "${HOMEPAGE_URL}"
EOF

RUN chmod +x /start.sh

CMD ["/start.sh"]