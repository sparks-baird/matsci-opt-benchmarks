If using Edison Scientific, you may need to wait up until 10 minutes to get the response. If you're an agent, sleep for 10 minutes (to avoid consuming requests unecessarily), and then every 5 minutes after that. Start with a 15 minute wait if using high effort literature query type. If you need to upload files, use analysis query type. See the docs: https://edisonscientific.gitbook.io/edison-cookbook/edison-client. Here is the endpoint you should use: https://api.platform.edisonscientific.com. The API key is EDISON_API_KEY. Don't expose this secret, e.g., by echoing or grepping it. Pass the API key in explicitly:

```
from edison_client import EdisonClient, JobNames
client = EdisonClient(api_key=EDISON_API_KEY)
```
