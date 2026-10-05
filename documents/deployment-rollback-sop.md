# Production Deployment Rollback SOP

A rollback is appropriate when a newly deployed version is strongly correlated with a customer-impacting regression and recovery through a safe configuration change is not faster.

## Pre-rollback checks
- Confirm the currently deployed image/version and the last known healthy version.
- Capture relevant application logs, Kubernetes events, and dashboard screenshots.
- Announce the rollback in the incident channel.
- Verify that database migrations are backward-compatible; if they are not, a rollback may require a separate database migration.

## Rollback procedure (Kubernetes)
1. Identify the previous revision: `kubectl rollout history deployment/<name> -n <namespace>`
2. Roll back to the previous revision: `kubectl rollout undo deployment/<name> -n <namespace>`
3. Watch the rollout: `kubectl rollout status deployment/<name> -n <namespace>`
4. Verify pod health: `kubectl get pods -n <namespace> -l app=<name>`
5. Confirm error rate drops to baseline in Grafana within 5 minutes.

## Rollback procedure (Helm)
1. List releases: `helm history <release-name> -n <namespace>`
2. Roll back: `helm rollback <release-name> <revision> -n <namespace>`
3. Confirm: `helm status <release-name> -n <namespace>`

## Post-rollback actions
- Update the incident channel with rollback completion.
- Preserve all logs and traces for post-incident review.
- Open a follow-up ticket to investigate the root cause before re-deploying.
- Re-enable any feature flags that were active before the rollback.

## When NOT to roll back
- When the root cause is a data migration already applied to the database.
- When rolling back would create a version mismatch with a dependent service already updated.
- When a hotfix is faster and safer than a full rollback.
