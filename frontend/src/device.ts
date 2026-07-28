type NavigatorDeviceInfo = Pick<
  Navigator,
  'maxTouchPoints' | 'platform' | 'userAgent'
> & {
  userAgentData?: {
    mobile?: boolean
  }
}

export function isIOSDevice(device: NavigatorDeviceInfo = navigator): boolean {
  return /iPad|iPhone|iPod/.test(device.userAgent)
    || (device.platform === 'MacIntel' && device.maxTouchPoints > 1)
}

export function isMobileDevice(device: NavigatorDeviceInfo = navigator): boolean {
  return device.userAgentData?.mobile === true
    || isIOSDevice(device)
    || /Android|webOS|BlackBerry|IEMobile|Opera Mini|Mobile/i.test(device.userAgent)
}
