# AWS Login & IAM Identity Center (SSO) Support Runbook

## Overview
This runbook covers authentication, corporate SSO login issues, permission set failures, and access denials for company AWS accounts managed via AWS IAM Identity Center (formerly AWS Single Sign-On).

## Common Symptoms
- "Access Denied" or "Invalid SSO token" when attempting to log in.
- Browser opens corporate IdP (Okta / Google Workspace / Azure AD) but returns redirect error or MFA timeout.
- CLI error: `Error loading SSO Token: The SSO session associated with this profile has expired`.
- AWS Console shows empty accounts list or "No access available".

## Triage & Resolution Procedures

### 1. Expired SSO Session Token (Most Common)
1. Clear stale cached session tokens from the local machine:
   `rm -rf ~/.aws/sso/cache/` (or `del /q %USERPROFILE%\.aws\sso\cache\*` on Windows).
2. Initiate a fresh CLI session login:
   `aws sso login --profile company-sso`
3. Complete MFA authentication in the company browser session.

### 2. Missing Account or Permission Set Assignment
1. Verify user is assigned to the target AWS account in AWS IAM Identity Center.
2. Confirm the corporate identity directory (Okta/Google/Azure AD) user group sync has not lagged.
3. If user changed teams or recently joined, contact the IT Cloud Admin team to assign the appropriate Permission Set (e.g. DeveloperAccess, ReadOnlyAccess).

### 3. Corporate VPN & IP Whitelisting
1. Ensure the device is connected to the company GlobalProtect / WireGuard VPN.
2. Some AWS accounts restrict console login to corporate egress IP addresses.

