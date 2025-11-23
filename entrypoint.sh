#!/bin/bash
set -e

# Export MCP server command location
export MCP_SERVER_COMMAND=/usr/local/bin/openstack-k8s-mcp

# Check for Kubernetes service account token and set KUBECONFIG if it exists
if [ -f /var/run/secrets/kubernetes.io/serviceaccount/token ]; then
    echo "Kubernetes service account token found, setting KUBECONFIG"
    export KUBECONFIG=/var/run/secrets/kubernetes.io/serviceaccount/token
fi

# Check for Kubernetes namespace and set NAMESPACE if it exists
if [ -f /var/run/secrets/kubernetes.io/serviceaccount/namespace ]; then
    echo "Kubernetes namespace file found, setting NAMESPACE"
    export NAMESPACE=$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace)
    echo "NAMESPACE set to: $NAMESPACE"
fi

# Launch the Python application
echo "Starting openstack-assistant..."
exec openstack-assistant "$@"
