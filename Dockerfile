# Use Ubuntu 26.04 as base image
FROM ubuntu:26.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies (using ubuntu-desktop-minimal for GNOME)
RUN apt-get update && \
    apt-get install -y \
    systemd \
    systemd-sysv \
    kmod \
    ethtool \
    ubuntu-desktop-minimal \
    iproute2 \
    openssh-server \
    wget \
    sudo \
    mesa-va-drivers \
    mesa-vulkan-drivers \
    $( [ "$(dpkg --print-architecture)" = "amd64" ] && echo intel-media-va-driver ) \
    vainfo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 1. Mask host-conflicting services:
# - NetworkManager: prevents hijacking host network interfaces
# - snapd: prevents container boot hanging
# - systemd-udev-trigger: prevents reloading host disk/network controllers
# - getty: prevents text login overriding the Wayland UI
RUN systemctl mask NetworkManager.service \
                   NetworkManager-wait-online.service \
                   snapd.service snapd.socket snapd.seeded.service \
                   systemd-udev-trigger.service \
                   getty@tty1.service \
                   systemd-vconsole-setup.service \
                   console-getty.service

# 2. Add our sandbox rules to ignore non-UI hardware
RUN mkdir -p /etc/udev/rules.d && \
    echo 'ACTION=="add|change", SUBSYSTEM!="drm|graphics|input|tty", OPTIONS:="ignore"' > /etc/udev/rules.d/00-container-sandbox.rules

# 3. Create a custom, safe trigger service that runs BEFORE logind
RUN echo "[Unit]\n\
Description=Targeted Container Udev Trigger\n\
Before=systemd-logind.service\n\
Requires=systemd-udevd.service\n\
After=systemd-udevd.service\n\
\n\
[Service]\n\
Type=oneshot\n\
# Only trigger UI hardware so we don't mess with host disk/network states\n\
ExecStart=/bin/udevadm trigger --action=add --subsystem-match=drm --subsystem-match=graphics --subsystem-match=input\n\
ExecStartPost=/bin/udevadm settle\n\
RemainAfterExit=yes\n\
\n\
[Install]\n\
WantedBy=multi-user.target\n" > /etc/systemd/system/container-udev-trigger.service && \
    systemctl enable container-udev-trigger.service

# CRITICAL FIX 1: Remove gnome-initial-setup. 
# This wizard runs on first login and notoriously crashes Wayland sessions in containers.
RUN apt-get remove --purge -y gnome-initial-setup && \
    apt-get autoremove -y

# CRITICAL FIX 2: Create User and bypass static ACL limitations
# Give both 'qd' and 'gdm' direct hardware access to GPUs and Input devices.
RUN useradd -m qd && \
    echo "qd:qdpassword" | chpasswd && \
    usermod -aG sudo,video,render,input qd && \
    usermod -aG video,render,input,tty gdm || true

# Set the default shell for qd to bash
RUN chsh -s /bin/bash qd

# Force GDM to Auto-Login user 'qd'. 
# This skips the buggy greeter screen and boots straight into the Wayland desktop.
RUN sed -i 's/#  AutomaticLoginEnable = true/AutomaticLoginEnable = True/' /etc/gdm3/custom.conf && \
    sed -i 's/#  AutomaticLogin = user1/AutomaticLogin = qd/' /etc/gdm3/custom.conf && \
    # Ensure Wayland is explicitly enabled (don't fallback to X11)
    sed -i 's/#WaylandEnable=false/WaylandEnable=true/' /etc/gdm3/custom.conf

# Set up SSH & Keys
RUN mkdir -p /var/run/sshd && \
    mkdir -p /home/qd/.ssh && \
    chown qd:qd /home/qd/.ssh && \
    chmod 700 /home/qd/.ssh
ADD https://github.com/danward.keys /home/qd/.ssh/authorized_keys
ADD https://github.com/alexanderturner.keys /home/qd/.ssh/authorized_keys_alex
RUN cat /home/qd/.ssh/authorized_keys_alex >> /home/qd/.ssh/authorized_keys && \
    rm /home/qd/.ssh/authorized_keys_alex && \
    chown qd:qd /home/qd/.ssh/authorized_keys && \
    chmod 600 /home/qd/.ssh/authorized_keys

# Disable GNOME screen timeout and sleep
RUN mkdir -p /etc/dconf/profile && \
    echo "user-db:user" > /etc/dconf/profile/user && \
    echo "system-db:local" >> /etc/dconf/profile/user && \
    mkdir -p /etc/dconf/db/local.d && \
    echo "[org/gnome/desktop/session]" > /etc/dconf/db/local.d/00-noblank && \
    echo "idle-delay=uint32 0" >> /etc/dconf/db/local.d/00-noblank && \
    echo "[org/gnome/desktop/screensaver]" >> /etc/dconf/db/local.d/00-noblank && \
    echo "lock-enabled=false" >> /etc/dconf/db/local.d/00-noblank && \
    dconf update

# Install Google Chrome (Avoiding Firefox because Ubuntu forces snapd for it, which we masked)
# Google only publishes Chrome debs for amd64, so skip the install on arm64
RUN if [ "$(dpkg --print-architecture)" = "amd64" ]; then \
        wget -q -O chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
        apt-get update && apt-get install -y ./chrome.deb && \
        rm chrome.deb; \
    fi && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the default entrypoint to systemd
STOPSIGNAL SIGRTMIN+3
CMD ["/lib/systemd/systemd"]