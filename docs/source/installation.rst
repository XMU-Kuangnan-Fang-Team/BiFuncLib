installation
=========================


Install from PyPI
------------------
It is recommended to use ``pip`` for installation. You need to check the upgrade at the beginning:

.. code-block:: bash
   
   pip install --upgrade pip

Then you can instll this package using the code:

.. code-block:: bash

   pip install BiFuncLib

You can also use the installation method for a specified version.

.. code-block:: bash

   pip install BiFuncLib==x.y.z

For faster installation, you can also use a mirror. For example, we use the Tsinghua mirror:

.. code-block:: bash

   pip install BiFuncLib==x.y.z -i https://pypi.tuna.tsinghua.edu.cn/simple

For detailed version information, please refer to `releases <https://github.com/XMU-Kuangnan-Fang-Team/BiFuncLib/releases>`_.

Install from source
---------------------

If you want to work with the latest development version or wish to contribute to **BiFuncLib**, this way will be suitable.

Clone the latest source code from GitHub and navigate into the directory of the cloned project:

.. code-block:: bash

   git clone https://github.com/XMU-Kuangnan-Fang-Team/BiFuncLib.git
   cd BiFuncLib

Run the following command to install the project:

.. code-block:: bash

   pip install .

Alternatively, if you want to install the project in development mode (which will install the project along with all development dependencies), you can use:

.. code-block:: bash

   pip install -e .

If you prefer to use the setup.py script to install, you can run:

.. code-block:: bash

   python setup.py install


Dependencies
---------------

Dependencies and version control for running **BiFuncLib**:

- genetlib>=1.2.4

- matplotlib>=3.7.1

- networkx>=2.8.4

- numpy==1.24.3

- pandas>=1.5.3

- scikit-learn>=1.2.2

- scikit-learn-extra==0.3.0

- scipy>=1.10.1

- seaborn>=0.13.2

- setuptools==67.8.0

When you install using the **pip install** method, these dependencies will be automatically installed.
