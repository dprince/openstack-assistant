"""OpenAPI tool definitions for the OpenStack Upgrade Assistant.

This module defines all available tools in OpenAPI format compatible with
Granite LLM models. Tools are converted from MCP tool definitions when needed.
"""

from typing import Any, Dict, List


def get_openstack_tools() -> List[Dict[str, Any]]:
    """Get OpenStack upgrade tools in OpenAPI format.

    Returns:
        List of tool definitions in OpenAPI function calling format
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_openstack_version",
                "description": (
                    "Query version status of the OpenStack deployment. "
                    "Returns deployedVersion (currently running), targetVersion (upgrading to), "
                    "and availableVersion (newest available), plus condition statuses."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'openstack')"
                        },
                        "name": {
                            "type": "string",
                            "description": "Name of OpenStackVersion resource (auto-discovers if not provided)"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "update_openstack_version",
                "description": (
                    "Initiate an OpenStack upgrade by updating the targetVersion. "
                    "This starts the minor update process, beginning with OVN update on Controlplane."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "targetVersion": {
                            "type": "string",
                            "description": "Target version to upgrade to (e.g., '0.0.2')"
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'openstack')"
                        },
                        "customContainerImages": {
                            "type": "object",
                            "description": "Optional custom container images to use for the upgrade"
                        }
                    },
                    "required": ["targetVersion"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "wait_openstack_version",
                "description": (
                    "Wait for a specific condition on OpenStackVersion resource to become true. "
                    "Automatically polls until condition is met or timeout is reached."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "condition": {
                            "type": "string",
                            "description": (
                                "Condition name to wait for (e.g., 'MinorUpdateOVNControlplane', "
                                "'MinorUpdateOVNDataplane', 'MinorUpdateControlplane', 'MinorUpdateDataplane')"
                            )
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'openstack')"
                        },
                        "name": {
                            "type": "string",
                            "description": "Name of OpenStackVersion resource (auto-discovers if not provided)"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 600)"
                        },
                        "pollInterval": {
                            "type": "integer",
                            "description": "Poll interval in seconds (default: 5)"
                        }
                    },
                    "required": ["condition"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "verify_openstack_controlplane",
                "description": (
                    "Check the health and status of the OpenStack controlplane. "
                    "Returns readiness, service health, and update status."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'openstack')"
                        },
                        "name": {
                            "type": "string",
                            "description": "Name of OpenStackControlplane resource (auto-discovers if not provided)"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_dataplane_deployment_ovn",
                "description": (
                    "Create an OpenStackDataplaneDeployment for OVN updates. "
                    "This is required during the OVN dataplane deployment phase of the upgrade."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Name for the deployment (e.g., 'edpm-deployment-ovn-update-0-0-2'). "
                                "Dots in version numbers should be converted to dashes."
                            )
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'openstack')"
                        },
                        "spec": {
                            "type": "object",
                            "description": "Full deployment spec as JSON (optional, uses defaults if not provided)"
                        }
                    },
                    "required": ["name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_dataplane_deployment_update",
                "description": (
                    "Create an OpenStackDataplaneDeployment for general dataplane updates. "
                    "This is required during the dataplane update phase of the upgrade."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Name for the deployment (e.g., 'edpm-deployment-update-0-0-2'). "
                                "Dots in version numbers should be converted to dashes."
                            )
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'openstack')"
                        },
                        "spec": {
                            "type": "object",
                            "description": "Full deployment spec as JSON (optional, uses defaults if not provided)"
                        }
                    },
                    "required": ["name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "verify_openstack_dataplanenodesets",
                "description": (
                    "Check the health and status of all OpenStack dataplane nodesets. "
                    "Returns health status, deployed images, and node states."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'openstack')"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_resume_step",
                "description": (
                    "Determine which upgrade step to resume from based on current OpenStack version state. "
                    "This tool analyzes targetVersion, availableVersion, and notReadyConditions to calculate "
                    "the exact step number where the upgrade should continue. Use this immediately after "
                    "get_openstack_version if an upgrade is already in progress (targetVersion == availableVersion). "
                    "Returns a step number (2-10) and explanation of why that step was chosen."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Kubernetes namespace (default: 'openstack')"
                        },
                        "name": {
                            "type": "string",
                            "description": "Name of OpenStackVersion resource (auto-discovers if not provided)"
                        }
                    },
                    "required": []
                }
            }
        }
    ]
