FROM consol/debian-xfce-vnc

# Switch to root user to install additional software
USER 0

## Install Pentobi
RUN apt update --assume-yes
RUN apt install --assume-yes flatpak
RUN flatpak remote-add --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo
RUN flatpak install -y flathub io.sourceforge.pentobi

# Start Pentobi on login.
RUN echo "flatpak run io.sourceforge.pentobi &" >> ~/.profile