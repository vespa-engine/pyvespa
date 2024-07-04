# Command line script to deploy Vespa applications to Vespa Cloud
# Usage: python -m vespa.utils.deploy_prod --tenant --application --api-key --application-root --max-wait 3600 --source-url

import argparse

from vespa.deployment import VespaCloud


def deploy_prod(
    tenant, application, api_key, application_root, max_wait=3600, source_url=None
):
    """
    Command-line script to deploy Vespa application to production in Vespa Cloud.

    Example usage:
    ```
    python -m vespa.utils.deploy_prod --tenant <tenant> --application <application> --api-key <api_key> --application-root <application_root> --max-wait 3600 --source-url <source_url>
    ```

    Args:
        tenant (str): Vespa Cloud tenant
        application (str): Vespa Cloud application
        api_key (str): Vespa Cloud Control-plane API key
        application_root (str): Path to the Vespa application root. If application is packaged with maven, this should refer to the generated target/application directory.
        max_wait (int, optional): Max wait time in seconds. Defaults to 3600. If set to -1, the script will return immediately after deployment is submitted.
        source_url (str, optional): Source URL (git commit URL) for the deployment. Defaults to None.


    """
    vespa_cloud = VespaCloud(
        tenant=tenant,
        application=application,
        key_content=api_key,
        application_root=application_root,
    )
    build_no = vespa_cloud.deploy_to_prod(
        instance="default", application_root=application_root, source_url=source_url
    )
    if max_wait == -1:
        print(f"Deployment submitted. Build number: {build_no}")
        return
    success = vespa_cloud.wait_for_prod_deployment(build_no, max_wait=max_wait)
    if not success:
        raise ValueError(
            f"Deployment failed to complete within {max_wait} seconds. Please check the Vespa Cloud console for more information."
        )
    else:
        print("Deployment completed successfully.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--tenant", required=True, help="Vespa Cloud tenant")
    args.add_argument("--application", required=True, help="Vespa Cloud application")
    args.add_argument(
        "--api-key", required=True, help="Vespa Cloud Control-plane API key"
    )
    args.add_argument(
        "--application-root", required=True, help="Path to the Vespa application root"
    )
    args.add_argument(
        "--max-wait",
        type=int,
        default=3600,
        help="Max wait time in seconds. -1 to return immediately after deployment is submitted.",
    )
    args.add_argument(
        "--source-url", help="Source URL (git commit URL) for the deployment"
    )

    args = args.parse_args()
    deploy_prod(
        args.tenant,
        args.application,
        args.api_key,
        args.application_root,
        args.max_wait,
        args.source_url,
    )
