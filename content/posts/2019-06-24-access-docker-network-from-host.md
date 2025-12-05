+++
title = 'Accessing a Docker container from host'
date = 2019-06-24T03:49:34-03:00
tags = ["docker"]
draft = false
+++

> This post was originally submitted to [StackOverflow](https://stackoverflow.com/a/56741737/957997).

If you need a quick workaround to access a container from your host:

1. First, [get the container IP](https://stackoverflow.com/q/17157721/957997):

```bash
$ docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' container_name_or_id

172.19.0.9
```

2. If you need to use the container name, add it to your `/etc/hosts`.

```
# /etc/hosts

172.19.0.9 container_name
```

