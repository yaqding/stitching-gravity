.. _classes:

Object-oriented code
####################

Creating bindings for a custom type
===================================

Let's now look at a more complex example where we'll create bindings for a
custom C++ data structure named ``Pet``. Its definition is given below:

.. code-block:: cpp

    struct Pet {
        Pet(const std::string &name) : name(name) { }
        void setName(const std::string &name_) { name = name_; }
        const std::string &getName() const { return name; }

        std::string name;
    };

The binding code for ``Pet`` looks as follows:

.. code-block:: cpp

    #include <pybind11/pybind11.h>

    namespace py = pybind11;

    PYBIND11_PLUGIN(example) {
        py::module m("example", "pybind11 example plugin");

        py::class_<Pet>(m, "Pet")
            .def(py::init<const std::string &>())
            .def("setName", &Pet::setName)
            .def("getName", &Pet::getName);

        return m.ptr();
    }

:class:`class_` creates bindings for a C++ *class* or *struct*-style data
structure. :func:`init` is a convenience function that takes the types of a
constructor's parameters as template arguments and wraps the corresponding
constructor (see the :ref:`custom_constructors` section for details). An
interactive Python session demonstrating this example is shown below:

.. code-block:: pycon

    % python
    >>> import example
    >>> p = example.Pet('Molly')
    >>> print(p)
    <example.Pet object at 0x10cd98060>
    >>> p.getName()
    u'Molly'
    >>> p.setName('Charly')
    >>> p.getName()
    u'Charly'

.. seealso::

    Static member functions can be bound in the same way using
    :func:`class_::def_static`.

Keyword and default arguments
=============================
It is possible to specify keyword and default arguments using the syntax
discussed in the previous chapter. Refer to the sections :ref:`keyword_args`
and :ref:`default_args` for details.

Binding lambda functions
========================

Note how ``print(p)`` produced a rather useless summary of our data structure in the example above:

.. code-block:: pycon

    >>> print(p)
    <example.Pet object at 0x10cd98060>

To address this, we could bind an utility function that returns a human-readable
summary to the special method slot named ``__repr__``. Unfortunately, there is no
suitable functionality in the ``Pet`` data structure, and it would be nice if
we did not have to change it. This can easily be accomplished by binding a
Lambda function instead:

.. code-block:: cpp

        py::class_<Pet>(m, "Pet")
            .def(py::init<const std::string &>())
            .def("setName", &Pet::setName)
            .def("getName", &Pet::getName)
            .def("__repr__",
                [](const Pet &a) {
                    return "<example.Pet named '" + a.name + "'>";
                }
            );

Both stateless [#f1]_ and stateful lambda closures are supported by pybind11.
With the above change, the same Python code now produces the following output:

.. code-block:: pycon

    >>> print(p)
    <example.Pet named 'Molly'>

.. [#f1] Stateless closures are those with an empty pair of brackets ``[]`` as the capture object.

.. _properties:

Instance and static fields
==========================

We can also directly expose the ``name`` field using the
:func:`class_::def_readwrite` method. A similar :func:`class_::def_readonly`
method also exists for ``const`` fields.

.. code-block:: cpp

        py::class_<Pet>(m, "Pet")
            .def(py::init<const std::string &>())
            .def_readwrite("name", &Pet::name)
            // ... remainder ...

This makes it possible to write

.. code-block:: pycon

    >>> p = example.Pet('Molly')
    >>> p.name
    u'Molly'
    >>> p.name = 'Charly'
    >>> p.name
    u'Charly'

Now suppose that ``Pet::name`` was a private internal variable
that can only be accessed via setters and getters.

.. code-block:: cpp

    class Pet {
    public:
        Pet(const std::string &name) : name(name) { }
        void setName(const std::string &name_) { name = name_; }
        const std::string &getName() const { return name; }
    private:
        std::string name;
    };

In this case, the method :func:`class_::def_property`
(:func:`class_::def_property_readonly` for read-only data) can be used to
provide a field-like interface within Python that will transparently call
the setter and getter functions:

.. code-block:: cpp

        py::class_<Pet>(m, "Pet")
            .def(py::init<const std::string &>())
            .def_property("name", &Pet::getName, &Pet::setName)
            // ... remainder ...

.. seealso::

    Similar functions :func:`class_::def_readwrite_static`,
    :func:`class_::def_readonly_static` :func:`class_::def_property_static`,
    and :func:`class_::def_property_readonly_static` are provided for binding
    static variables and properties. Please also see the section on
    :ref:`static_properties` in the advanced part of the documentation.

Dynamic attributes
==================

Native Python classes can pick up new attributes dynamically:

.. code-block:: pycon

    >>> class Pet:
    ...     name = 'Molly'
    ...
    >>> p = Pet()
    >>> p.name = 'Charly'  # overwrite existing
    >>> p.age = 2  # dynamically add a new attribute

By default, classes exported from C++ do not support this and the only writable
attributes are the ones explicitly defined using :func:`class_::def_readwrite`
or :func:`class_::def_property`.

.. code-block:: cpp

    py::class_<Pet>(m, "Pet")
        .def(py::init<>())
        .def_readwrite("name", &Pet::name);

Trying to set any other attribute results in an error:

.. code-block:: pycon

    >>> p = example.Pet()
    >>> p.name = 'Charly'  # OK, attribute defined in C++
    >>> p.age = 2  # fail
    AttributeError: 'Pet' object has no attribute 'age'

To enable dynamic attributes for C++ classes, the :class:`py::dynamic_attr` tag
must be added to the :class:`py::class_` constructor:

.. code-block:: cpp

    py::class_<Pet>(m, "Pet", py::dynamic_attr())
        .def(py::init<>())
        .def_readwrite("name", &Pet::name);

Now everything works as expected:

.. code-block:: pycon

    >>> p = example.Pet()
    >>> p.name = 'Charly'  # OK, overwrite value in C++
    >>> p.age = 2  # OK, dynamically add a new attribute
    >>> p.__dict__  # just like a native Python class
    {'age': 2}

Note that there is a small runtime cost for a class with dynamic attributes.
Not only because of the addition of a ``__dict__``, but also because of more
expensive garbage collection tracking which must be activated to resolve
possible circular references. Native Python classes incur this same cost by
default, so this is not anything to worry about. By default, pybind11 classes
are more efficient than native Python classes. Enabling dynamic attributes
just brings them on par.

.. _inheritance:

Inheritance
===========

Suppose now that the example consists of two data structures with an
inheritance relationship:

.. code-block:: cpp

    struct Pet {
        Pet(const std::string &name) : name(name) { }
        std::string name;
    };

    struct Dog : Pet {
        Dog(const std::string &name) : Pet(name) { }
        std::string bark() const { return "woof!"; }
    };

There are two different ways of indicating a hierarchical relationship to
pybind11: the first specifies the C++ base class as an extra template
parameter of the :class:`class_`:

.. code-block:: cpp

    py::class_<Pet>(m, "Pet")
       .def(py::init<const std::string &>())
       .def_readwrite("name", &Pet::name);

    // Method 1: template parameter:
    py::class_<Dog, Pet /* <- specify C++ parent type */>(m, "Dog")
        .def(py::init<const std::string &>())
        .def("bark", &Dog::bark);

Alternatively, we can also assign a name to the previously bound ``Pet``
:class:`class_` object and reference it when binding the ``Dog`` class:

.. code-block:: cpp

    py::class_<Pet> pet(m, "Pet");
    pet.def(py::init<const std::string &>())
       .def_readwrite("name", &Pet::name);

    // Method 2: pass parent class_ object:
    py::class_<Dog>(m, "Dog", pet /* <- specify Python parent type */)
        .def(py::init<const std::string &>())
        .def("bark", &Dog::bark);

Functionality-wise, both approaches are equivalent. Afterwards, instances will
expose fields and methods of both types:

.. code-block:: pycon

    >>> p = example.Dog('Molly')
    >>> p.name
    u'Molly'
    >>> p.bark()
    u'woof!'

Overloaded methods
==================

Sometimes there are several overloaded C++ methods with the same name taking
different kinds of input arguments:

.. code-block:: cpp

    struct Pet {
        Pet(const std::string &name, int age) : name(name), age(age) { }

        void set(int age_) { age = age_; }
        void set(const std::string &name_) { name = name_; }

        std::string name;
        int age;
    };

Attempting to bind ``Pet::set`` will cause an error since the compiler does not
know which method the user intended to select. We can disambiguate by casting
them to function pointers. Binding multiple functions to the same Python name
automatically creates a chain of function overloads that will be tried in
sequence.

.. code-block:: cpp

    py::class_<Pet>(m, "Pet")
       .def(py::init<const std::string &, int>())
       .def("set", (void (Pet::*)(int)) &Pet::set, "Set the pet's age")
       .def("set", (void (Pet::*)(const std::string &)) &Pet::set, "Set the pet's name");

The overload signatures are also visible in the method's docstring:

.. code-block:: pycon

    >>> help(example.Pet)

    class Pet(__builtin__.object)
     |  Methods defined here:
     |
     |  __init__(...)
     |      Signature : (Pet, str, int) -> NoneType
     |
     |  set(...)
     |      1. Signature : (Pet, int) -> NoneType
     |
     |      Set the pet's age
     |
     |      2. Signature : (Pet, str) -> NoneType
     |
     |      Set the pet's name

If you have a C++14 compatible compiler [#cpp14]_, you can use an alternative
syntax to cast the overloaded function:

.. code-block:: cpp

    py::class_<Pet>(m, "Pet")
        .def("set", py::overload_cast<int>(&Pet::set), "Set the pet's age")
        .def("set", py::overload_cast<const std::string &>(&Pet::set), "Set the pet's name");

Here, ``py::overload_cast`` only requires the parameter types to be specified.
The return type and class are deduced. This avoids the additional noise of
``void (Pet::*)()`` as seen in the raw cast. If a function is overloaded based
on constness, the ``py::const_`` tag should be used:

.. code-block:: cpp

    struct Widget {
        int foo(int x, float y);
        int foo(int x, float y) const;
    };

    py::class_<Widget>(m, "Widget")
       .def("foo_mutable", py::overload_cast<int, float>(&Widget::foo))
       .def("foo_const",   py::overload_cast<int, float>(&Widget::foo, py::const_));


.. [#cpp14] A compiler which supports the ``-std=c++14`` flag
            or Visual Studio 2015 Update 2 and newer.

.. note::

    To define multiple overloaded constructors, simply declare one after the
    other using the ``.def(py::init<...>())`` syntax. The existing machinery
    for specifying keyword and default arguments also works.

Enumerations and internal types
===============================

Let's now suppose that the example class contains an internal enumeration type,
e.g.:

.. code-block:: cpp

    struct Pet {
        enum Kind {
            Dog = 0,
            Cat
        };

        Pet(const std::string &name, Kind type) : name(name), type(type) { }

        std::string name;
        Kind type;
    };

The binding code for this example looks as follows:

.. code-block:: cpp

    py::class_<Pet> pet(m, "Pet");

    pet.def(py::init<const std::string &, Pet::Kind>())
        .def_readwrite("name", &Pet::name)
        .def_readwrite("type", &Pet::type);

    py::enum_<Pet::Kind>(pet, "Kind")
        .value("Dog", Pet::Kind::Dog)
        .value("Cat", Pet::Kind::Cat)
        .export_values();

To ensure that the ``Kind`` type is created within the scope of ``Pet``, the
``pet`` :class:`class_` instance must be supplied to the :class:`enum_`.
constructor. The :func:`enum_::export_values` function exports the enum entries
into the parent scope, which should be skipped for newer C++11-style strongly
typed enums.

.. code-block:: pycon

    >>> p = Pet('Lucy', Pet.Cat)
    >>> p.type
    Kind.Cat
    >>> int(p.type)
    1L

The entries defined by the enumeration type are exposed in the ``__members__`` property:

.. code-block:: pycon

    >>> Pet.Kind.__members__
    {'Dog': Kind.Dog, 'Cat': Kind.Cat}

.. note::

    When the special tag ``py::arithmetic()`` is specified to the ``enum_``
    constructor, pybind11 creates an enumeration that also supports rudimentary
    arithmetic and bit-level operations like comparisons, and, or, xor, negation,
    etc.

    .. code-block:: cpp

        py::enum_<Pet::Kind>(pet, "Kind", py::arithmetic())
           ...

    By default, these are omitted to conserve space.
