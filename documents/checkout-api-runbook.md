# Checkout API Production Runbook

## Service overview
The Checkout API receives checkout requests from the web application and communicates with the Payment Gateway, Inventory Service, Redis session cache, and PostgreSQL Orders database. Production traffic enters through the public load balancer and is forwarded to the checkout-api Kubernetes deployment.

## 502 errors after a deployment
When 502 responses begin immediately after a release, the on-call engineer should use the following order of checks:

1. Compare the incident start time with the latest deployment timestamp.
2. Check Kubernetes deployment and pod status for `checkout-api`.
3. Inspect readiness-probe failures before restarting pods. A pod that fails readiness must not receive production traffic.
4. Review the latest checkout-api container logs for startup exceptions or missing environment variables.
5. Confirm the new image digest is present on all nodes by checking the pod spec.
6. If all pods are failing, roll back to the previous image tag immediately using the Deployment Rollback SOP.

## 503 / timeout errors under load
1. Check HPA (HorizontalPodAutoscaler) — confirm autoscaling is not stuck at minReplicas.
2. Review Redis latency; checkout-api uses Redis for session tokens and cart state.
3. Verify the Payment Gateway is healthy — its /health endpoint should return 200.
4. Check PostgreSQL connection pool saturation via pg_stat_activity.

## Database connection failures
- Error pattern: `FATAL: remaining connection slots are reserved`
- Action: check current connection count, kill idle connections older than 10 minutes, and increase max_connections if recurring.

## High error rate on /checkout endpoint
- PagerDuty threshold: >5% 5xx for 5 minutes.
- Confirm Payment Gateway, Inventory Service, and Redis are all healthy before assuming a code defect.

## Rollback decision
If the error rate does not drop within 15 minutes of investigation, initiate rollback per the Deployment Rollback SOP.
