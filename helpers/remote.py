import json

import paramiko

from helpers.config import config


class Remote:
    def __init__(self):
        with open(config.sshCredsFile) as json_file:
            sshCreds = json.load(json_file)

        host = sshCreds['host']
        port = sshCreds['port']
        password = sshCreds['pass']
        username = sshCreds['user']

        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(host, port, username, password)
        self.ssh.exec_command("mkdir -p " + config.remoteDir)

    def upload(self, localFile, remoteFile):
        sftp = self.ssh.open_sftp()
        sftp.put(localFile, remoteFile)
        sftp.close()
