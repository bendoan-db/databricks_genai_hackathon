%sh
#!/bin/bash

# Define path for requirements.txt file
# REQUIREMENTS_FILE_PATH="/databricks/driver/requirements.txt"

# Create folder for requirements.txt if it doesn't exist
if [ ! -d /tmp/felix.flory@databricks.com ]; then
  mkdir /tmp/felix.flory@databricks.com
fi

REQUIREMENTS_FILE_PATH="/tmp/felix.flory@databricks.com/requirements.txt"

# Create 'packages' directory if it doesn't exist
if [ ! -d /tmp/felix.flory@databricks.com/packages ]; then
  mkdir /tmp/felix.flory@databricks.com/packages
fi


# Create and open the file in write mode
cat > $REQUIREMENTS_FILE_PATH <<EOF
mlflow==2.21.3
langgraph==0.3.4
uv==0.6.10
databricks-agents==0.18.1
databricks-langchain==0.4.1
databricks-vectorsearch==0.55
unitycatalog-langchain==0.2.0
databricks-sqlalchemy==2.0.5
EOF

echo "File 'requirements.txt' created successfully at $REQUIREMENTS_FILE_PATH."

# Download packages to 'packages' directory
pip download -r $REQUIREMENTS_FILE_PATH -d /tmp/felix.flory@databricks.com/packages/

# Create tar file
tar cvfz /tmp/felix.flory@databricks.com/dependencies.tar.gz /tmp/felix.flory@databricks.com/packages

echo "Packages downloaded successfully"