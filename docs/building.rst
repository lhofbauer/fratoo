
================
Building a model
================

A fratoo model consists of two elements: a file describing the model equations and an input data set.

***************
Model equations
***************

The first step in building a fratoo model is establishing the model equations by setting up an OSeMOSYS (Pyomo-based) model file, i.e., a Python file defining a `Pyomo <http://http://www.pyomo.org/>`_  abstract model. A model file can be downloaded from the `OSeMOSYS github page <https://github.com/OSeMOSYS/OSeMOSYS>`_ and either be used directly or amended in order to add/alter functionality. 

A detailed description of the standard OSeMOSYS model file and its sets, parameters, variables, and equations can be found in the `OSeMOSYS documentation <https://osemosys.readthedocs.io/en/latest/manual/Structure%20of%20OSeMOSYS.html>`_. A detailed description of Pyomo and how to build Pyomo models can be found `here <https://pyomo.readthedocs.io/en/stable/>`_.




**********
Input data 
**********

After defining the model structure, the second part of building a fratoo model is setting up the input data, e.g., capital cost of technologies. The sets and parameters to be defined in the input data depend on the model structure, i.e., model equations. A description of the sets and parameters to be defined for a standard OSeMOSYS model (and thus for a fratoo model using the standard OSeMOSYS code) can be found in the `OSeMOSYS docs <https://osemosys.readthedocs.io/en/latest/manual/Structure%20of%20OSeMOSYS.html>`_.

fratoo introduces 3 major additions/changes to the 'standard' OSeMOSYS input data set:

#. It introduces a **syntax** for abbreviations to allow for an easier definition of models.
#. It adds a **multi-scale structure** to the model geography based on 4 additional multi-scale parameters.
#. It changes the way **spatial entities** are used within the input data set.

These specific features are further explained below, after introducing the general data format used for fratoo.

-----------
Data format
-----------

The framework requires input data in the form of a `frictionless data package <https://github.com/frictionlessdata/datapackage-py>`_, which is also used by the OSeMOSYS toolbox `otoole <https://github.com/OSeMOSYS/otoole>`_. An example data package is provided in an `OSeMOSYS github repository <https://github.com/OSeMOSYS/simplicity>`_.

.. add link to fratoo example data package

The data package consist of a

* licence file,
* JSON metadata file, and
* the data files in the data directory.

The data are a set of CSV files, one for each OSeMOSYS parameter and set. As such, fratoo can be used with standard OSeMOSYS input data sets. If a multi-scale model is to be built, 4 additional files are necessary, which are further explained below.


CSV files defining parameters or sets are defined in a simple tabular manner, as shown in the exemplary tables below.

.. csv-table:: Parameter: operational life | OperationalLife.csv
   :file: figures/OperationalLife.csv
   :widths: 33,33,33
   :header-rows: 1

.. csv-table:: Set: regions | REGION.csv
   :file: figures/REGION.csv
   :widths: 100
   :header-rows: 1




-----------------
Model geography
-----------------

As normal OSeMOSYS models, fratoo allows for the design of multi-regional models. In addition, fratoo introduces an explicit multi-scale geography of its spatial entities (for the general concept refer to the :doc:`introduction <introduction>`).

The framework allows the modeller to specify an arbitrary number of spatial entities, e.g., a set of local areas and the entire country. Spatial entities are assigned a particular scale and, apart from the overarching spatial entity, a so called *parent entity* on an upper scale, i.e., the entity it belongs to (e.g., a local area on the local scale is part of a particular country on the national scale).

This structure is defined through 4 additional CSV files (marked with the file prefix \'ft\_\') with the same format as the standard OSeMOSYS parameter files. *ft_scale.csv* assigns a scale to each spatial entity, *ft_affiliation.csv* assigns a parent entity, and *ft_param_agg.csv* and *ft_param_disagg.csv* define how parameters are processed if entities need to be (dis)aggregated for a model run (further explained in the :doc:`running` section). Exemplary files for scale and affiliation assignment are shown below.

.. csv-table:: Scale | ft_scale.csv
   :file: figures/ft_scale.csv
   :widths: 50,50
   :header-rows: 1

.. csv-table:: Affiliation/parent entities | ft_affiliation.csv
   :file: figures/ft_affiliation.csv
   :widths: 50,50
   :header-rows: 1

This approach allows to build models of a particular geographic area, e.g., a country, while explicitly representing subareas, e.g., regions or local areas. Thus, different elements of the energy system can be modelled at different spatial scales based on the required detail or available data. For example, a transport sector model looking at the transportation sector in all local areas of a country might incorporate a simple version of a power sector, defined at the national scale, providing power to all local areas.

This structure comes into play when specifying OSeMOSYS parameters that vary across spatial entities, i.e., that are defined over the set *REGION* as, for example, capital cost. These parameters are then, similarly to a normal OSeMOSYS model, defined for a specific geographic entity on the particular scale. The following table shows an example.

.. csv-table:: Parameter: capital cost | CaptialCost.csv
   :file: figures/CapitalCost.csv
   :header-rows: 1

As shown above, fratoo makes use of the OSeMOSYS set *REGION* for defining model inputs, yet, it does not actually use the inter-region functionality as implemented in the standard OSeMOSYS code as it restricts the way interactions between regions can be modelled. Instead, interactions between geographic entities can be implemented by introducing respective transport technologies and setting the input and output fuels to the respective spatial entities. In order to allow for technologies to set input/output fuels to another region, a new syntax is introduced as shown in the table below and further explained in the next subsection.

.. csv-table:: Parameter: input activity ratio | InputActivityRatio.csv
   :file: figures/InputActivityRatio.csv
   :widths: 16,16,16,16,16,16
   :header-rows: 1


.. set region should include same regions as ft_scale/affiliation
   an example input data package can be found here=link



----------------
Syntax additions
----------------

fratoo introduces a few additional syntax elements, which can be used in the input data set, i.e., CSV files, when specifying sets and parameters. These are listed below:

* **:\*** can be used to represent all values in the particular set
* **:\*[scale]** can be used to represent all spatial entities on the scale [scale]
* **:[spatial entity]:[fuel]** can be used to refer to the fuel [fuel] in the entity [spatial entity]
* **:[scale]:[fuel]** can be used to refer to the fuel [fuel] in the (grand/etc.) parent entity on scale [scale]

An example is shown below. It defines the input activity ratio for transformers in all local areas (:\*2) for all years (:\*). The transformers get electricity either from the UK entity/national scale (:UK:EL) or from the respective region of the particular local area (:1:EL), e.g., England for Brighton and Scotland for Edinburgh.

.. csv-table:: Parameter: input activity ratio | InputActivityRatio.csv
   :file: figures/InputActivityRatio_syntax.csv
   :widths: 16,16,16,16,16,16
   :header-rows: 1




