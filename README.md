# databricks_genai_hackathon

# Cluster Config

On Databricks use a serverless cluster or a standard cluster with runtime 15.4 LTS or higher. A Machine Learning Runtime is not recommended. 
For a standard databricks runtime please [install](https://docs.databricks.com/aws/en/libraries/cluster-libraries) the required libraries listed in [requirements.txt](requirements.txt). You can then ommit the ```pip install ...``` commands at the begging of the notbooks.

# For admins

- ideally, hackathon users should have permissions to create their individual unity catalog schema. This greatly reduces the need to specify individual assests like tablenames, uc-function names, models etc. 
- yaml files do not get cloned TODO specify defaults in the pydantic model itself and users can edit those defaults
- TODO if no internet access the tables have to be setup differently (load local parquet files to pandas and write to delta table)

# Project Setup

 - edit [configs/project.yml](configs/project.yml) to specify your settings
 - run the project setup notebook  [setup_env/workspace_assets.ipynb](setup_env/workspace_assets.ipynb)

 # Notebooks

After the project setup you can work through the notebooks in the notebooks folder. 


# Disclaimer

Disclaimer

These examples are provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors, copyright holders, or contributors be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

The authors and maintainers of this repository make no guarantees about the suitability, reliability, availability, timeliness, security or accuracy of the software. It is your responsibility to determine that the software meets your needs and complies with your system requirements.

No support is provided with this software. Users are solely responsible for installation, use, and troubleshooting. While issues and pull requests may be submitted, there is no guarantee of response or resolution.

By using this software, you acknowledge that you have read this disclaimer, understand it, and agree to be bound by its terms.
